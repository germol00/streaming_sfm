import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from openai import OpenAI
from simulstream.server.speech_processors import SAMPLE_RATE, SpeechProcessor
from simulstream.server.speech_processors.incremental_output import IncrementalOutput
from vllm import LLM, SamplingParams

from streaming_sfm.hyp_utils import (
    HoldNHypothesisBuffer,
    LACPHypothesisBuffer,
    LCPHypothesisBuffer,
    WaitKHypothesisBuffer,
)
from streaming_sfm.parakeet import _build_slcp_buffer
from streaming_sfm.streaming_model import (
    StreamingBatchedAudioBufferWithOffset,
    StreamingParakeet,
)

logger = logging.getLogger(__name__)
logging.getLogger("fbk_fairseq.simultaneous.metrics").setLevel(logging.INFO)


def longest_common_prefix(s1: str, s2: str) -> str:
    for i in range(min(len(s1), len(s2))):
        if s1[i] != s2[i]:
            return s1[:i]
    return s1[: min(len(s1), len(s2))]


@dataclass
class CascadeState:
    speech_id: int = 0
    asr_committed_text: str = ""
    prev_translation: str = ""
    translation_hypotheses: List[str] = field(default_factory=lambda: [""])
    emission_started: bool = False
    total_samples: int = 0

    # Streaming ASR internals
    asr_buffer: Optional[StreamingBatchedAudioBufferWithOffset] = None
    asr_hyp_buffer: Optional[object] = None
    current_offset: int = 0
    nchunks_no_output: int = 0


class CascadeSpeechProcessor(SpeechProcessor):
    """
    SimulStream processor with:
    - ASR: Streaming SFM + Parakeet
    - MT: Qwen-3.5 via vLLM (local or OpenAI-compatible endpoint)
    """

    @classmethod
    def load_model(cls, config: SimpleNamespace):
        if not hasattr(cls, "asr") or cls.asr is None:
            beam_size = getattr(config, "sfm_decode", 1)
            boosting_alpha = getattr(config, "sfm_boosting_tree_alpha", 1.0)
            boosting_cfg = {
                "context_score": getattr(config, "sfm_context_score", 1.0),
                "depth_scaling": getattr(config, "sfm_depth_scaling", 2.0),
            }
            decoding_cfg = {
                "strategy": "greedy_batch" if beam_size == 1 else "malsd_batch",
                "greedy": {
                    "boosting_tree": boosting_cfg,
                    "boosting_tree_alpha": boosting_alpha,
                },
                "beam": {
                    "boosting_tree": boosting_cfg,
                    "boosting_tree_alpha": boosting_alpha,
                    "beam_size": beam_size,
                },
            }

            cfg_args = SimpleNamespace(
                model_path=getattr(config, "sfm_model_path", None),
                pretrained_name=getattr(config, "sfm_pretrained_name", "nvidia/parakeet-tdt-0.6b-v3"),
                manifest_path=getattr(config, "sfm_manifest_path", "vp.jsonl"),
                chunk_secs=getattr(config, "sfm_chunk_secs", 1.0),
                left_context_secs=getattr(config, "sfm_left_context_secs", 20.0),
                right_context_secs=getattr(config, "sfm_right_context_secs", 0.0),
                max_empty_chunks=getattr(config, "sfm_max_empty_chunks", 0),
                policy=getattr(config, "sfm_policy", "LACP"),
                lacp_threshold=getattr(config, "sfm_lacp_threshold", 2.0),
                K=getattr(config, "sfm_K", 2),
                N=getattr(config, "sfm_N", 5),
                word_level=getattr(config, "sfm_word_level", False),
                slcp_semantic_threshold=getattr(config, "sfm_slcp_semantic_threshold", 0.65),
                slcp_max_gap=getattr(config, "sfm_slcp_max_gap", 3),
                slcp_use_spacy=getattr(config, "sfm_slcp_use_spacy", False),
                device=getattr(config, "sfm_device", "cuda"),
                compute_dtype=getattr(config, "sfm_compute_dtype", "bfloat16"),
                emit_incomplete=getattr(config, "sfm_emit_incomplete", False),
                rnnt_decoding=decoding_cfg,
            )
            asr_cfg = OmegaConf.create(vars(cfg_args))
            with open_dict(asr_cfg):
                asr_cfg.cuda = 0 if cfg_args.device == "cuda" else -1
                asr_cfg.allow_mps = True if cfg_args.device == "mps" else False
            cls.asr_cfg = asr_cfg
            cls.asr = StreamingParakeet(asr_cfg, mbr=getattr(config, "sfm_mbr", False))

        llm_base_url = getattr(config, "llm_base_url", None)
        if llm_base_url is not None:
            if not hasattr(cls, "llm_client") or cls.llm_client is None:
                cls.llm_client = OpenAI(base_url=llm_base_url, api_key="EMPTY")
                from transformers import AutoTokenizer

                cls.tokenizer = AutoTokenizer.from_pretrained(config.llm_model_name)
            cls.llm = None
        else:
            cls.llm_client = None
            if not hasattr(cls, "llm") or cls.llm is None:
                cls.llm = LLM(
                    model=config.llm_model_name,
                    trust_remote_code=True,
                    gpu_memory_utilization=getattr(config, "llm_gpu_memory_utilization", 0.75),
                    tensor_parallel_size=getattr(config, "llm_tensor_parallel_size", 1),
                    max_num_seqs=1,
                    max_model_len=getattr(config, "llm_max_model_len", 2048),
                    enable_prefix_caching=True,
                )
                cls.tokenizer = cls.llm.get_tokenizer()

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.load_model(config)

        self.source_lang = getattr(config, "source_lang", "English")
        self.target_lang = getattr(config, "target_lang", "German")
        self.target_sep = "" if self.target_lang in ["Chinese", "Japanese"] else " "
        self.latency_unit = getattr(config, "latency_unit", "word")

        self.min_start_seconds = getattr(config, "min_start_seconds", 1.0)
        self._temperature = getattr(config, "temperature", 0.7)
        self._top_p = getattr(config, "top_p", 0.9)
        self._top_k = getattr(config, "top_k", 20)
        self._max_tokens = getattr(config, "max_new_tokens", 256)
        self._repetition_penalty = getattr(config, "repetition_penalty", 1.05)
        self._llm_model_name = config.llm_model_name

        self.sampling_params = SamplingParams(
            temperature=self._temperature,
            top_p=self._top_p,
            top_k=self._top_k,
            max_tokens=self._max_tokens,
            repetition_penalty=self._repetition_penalty,
            stop=["\n"],
        )

        self._state = self._fresh_state(speech_id=0)

    def _fresh_state(self, speech_id: int) -> CascadeState:
        asr_buffer = StreamingBatchedAudioBufferWithOffset(
            batch_size=1,
            context_samples=self.asr.context_samples,
            dtype=self.asr.dtype,
            device=self.asr.device,
        )
        asr_hyp_buffer = self._build_asr_hyp_buffer()
        return CascadeState(
            speech_id=speech_id,
            asr_buffer=asr_buffer,
            asr_hyp_buffer=asr_hyp_buffer,
        )

    def _build_asr_hyp_buffer(self):
        cfg = self.asr_cfg
        word_level = getattr(cfg, "word_level", False)
        if cfg.policy == "LCP":
            return LCPHypothesisBuffer(word_level=word_level, debug=False)
        if cfg.policy == "LACP":
            return LACPHypothesisBuffer(cfg.lacp_threshold, word_level=word_level, debug=False)
        if cfg.policy == "SLCP":
            return _build_slcp_buffer(cfg, word_level=word_level, debug=False)
        if cfg.policy == "WaitK":
            return WaitKHypothesisBuffer(
                cfg.K,
                features_per_second=self.asr.features_per_sec,
                subsampling_factor=self.asr.subsampling_factor,
                word_level=word_level,
                debug=False,
            )
        return HoldNHypothesisBuffer(cfg.N, word_level=word_level, debug=False)

    def _tokens_to_text(self, toks: List[str]) -> str:
        word_level = getattr(self.asr_cfg, "word_level", False)
        if word_level:
            return "".join(t.replace("▁", " ") for t in toks).strip()
        return self.asr.asr_model.tokenizer.tokens_to_text(toks)

    def _asr_step(self, state: CascadeState, waveform: np.ndarray, is_last_chunk: bool) -> str:
        if (waveform is None or len(waveform) == 0) and not is_last_chunk:
            return ""

        if waveform is not None and len(waveform) > 0:
            chunk = np.asarray(waveform, dtype=np.float32)
            chunk_t = torch.tensor([chunk], device=self.asr.device)
            stride = state.asr_buffer.add_audio_batch_get_stride(
                chunk_t,
                audio_lengths=torch.tensor([len(chunk)], device=self.asr.device),
                is_last_chunk=is_last_chunk,
                is_last_chunk_batch=torch.tensor([is_last_chunk], device=self.asr.device),
            )
            state.current_offset += stride // self.asr.encoder_frame2audio_samples

            hyp = self.asr.process_chunk(state.asr_buffer, state.current_offset)
            state.asr_hyp_buffer.insert(hyp)

        max_empty_chunks = getattr(self.asr_cfg, "max_empty_chunks", 0)
        if self.asr_cfg.policy == "WaitK":
            out = state.asr_hyp_buffer.flush(last_instant=0)
        elif self.asr_cfg.policy == "LACP":
            out = state.asr_hyp_buffer.flush(forced=True)
        elif self.asr_cfg.policy == "LCP" and state.nchunks_no_output >= max_empty_chunks:
            out = state.asr_hyp_buffer.flush(forced=True)
        else:
            out = state.asr_hyp_buffer.flush()

        if is_last_chunk:
            out.extend(state.asr_hyp_buffer.complete())

        if max_empty_chunks:
            if not out:
                state.nchunks_no_output += 1
            else:
                state.nchunks_no_output = 0

        if not out:
            return ""
        toks = [t for _, _, t in out]
        return self._tokens_to_text(toks)

    def _prepare_llm_inputs(self, asr_segment: str, prev_translation: str) -> str:
        instruction = f"""You are a professional simultaneous speech translator.

[TASK]
Translate the input text from {self.source_lang} into {self.target_lang}.
Preserve named entities exactly as in the source text.
Return only the translated text, with no explanation.

[INPUT]
{asr_segment}"""
        messages = [{"role": "user", "content": instruction}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        return prompt + prev_translation

    def _llm_generate(self, prompt: str) -> str:
        if self.llm_client is not None:
            response = self.llm_client.completions.create(
                model=self._llm_model_name,
                prompt=prompt,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                top_p=self._top_p,
                stop=["\n"],
                extra_body={"repetition_penalty": self._repetition_penalty},
            )
            return response.choices[0].text.replace("…", "")
        llm_outputs = self.llm.generate(
            [prompt],
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        return llm_outputs[0].outputs[0].text.replace("…", "")

    def _translate_from_asr(self, state: CascadeState, force_final: bool) -> str:
        asr_text = state.asr_committed_text.strip()
        if not asr_text:
            return ""

        prompt = self._prepare_llm_inputs(asr_text, state.prev_translation)
        hypothesis = self._llm_generate(prompt)
        full_hypothesis = state.prev_translation + hypothesis

        if force_final:
            increment = full_hypothesis[len(state.prev_translation) :]
            state.prev_translation = full_hypothesis
            return increment.strip() if self.target_lang not in ["Chinese", "Japanese"] else increment

        state.translation_hypotheses.append(full_hypothesis)
        stable = longest_common_prefix(
            state.translation_hypotheses[-2],
            state.translation_hypotheses[-1],
        )
        increment = stable[len(state.prev_translation) :]
        state.prev_translation = stable
        if self.target_lang not in ["Chinese", "Japanese"]:
            increment = increment.strip()
        return increment

    def _text_to_tokens(self, text: str) -> List[str]:
        if text == "":
            return []
        if self.latency_unit in ["word", "spm"]:
            return text.strip().split()
        if self.latency_unit == "char":
            return list(text.strip())
        raise NotImplementedError(f"Unsupported latency_unit: {self.latency_unit}")

    def _build_incremental_output(self, text: str) -> IncrementalOutput:
        if text == "":
            return IncrementalOutput([], "", [], "")

        out_text = text
        if self.latency_unit == "word" and self._state.emission_started and not out_text.startswith(" "):
            out_text = " " + out_text
        self._state.emission_started = True

        return IncrementalOutput(
            new_tokens=self._text_to_tokens(text),
            new_string=out_text,
            deleted_tokens=[],
            deleted_string="",
        )

    @torch.inference_mode()
    def process_chunk(self, waveform: np.float32) -> IncrementalOutput:
        if waveform is None or len(waveform) == 0:
            return IncrementalOutput([], "", [], "")

        self._state.total_samples += len(waveform)
        total_duration = self._state.total_samples / SAMPLE_RATE
        if total_duration < self.min_start_seconds and self._state.asr_committed_text == "":
            return IncrementalOutput([], "", [], "")

        asr_increment = self._asr_step(self._state, waveform, is_last_chunk=False)
        if asr_increment:
            if self._state.asr_committed_text:
                self._state.asr_committed_text = f"{self._state.asr_committed_text} {asr_increment}".strip()
            else:
                self._state.asr_committed_text = asr_increment.strip()

        translation_increment = self._translate_from_asr(self._state, force_final=False)
        return self._build_incremental_output(translation_increment)

    @torch.inference_mode()
    def end_of_stream(self) -> IncrementalOutput:
        asr_increment = self._asr_step(self._state, np.zeros(0, dtype=np.float32), is_last_chunk=True)
        if asr_increment:
            if self._state.asr_committed_text:
                self._state.asr_committed_text = f"{self._state.asr_committed_text} {asr_increment}".strip()
            else:
                self._state.asr_committed_text = asr_increment.strip()

        translation = self._translate_from_asr(self._state, force_final=True)
        current_speech_id = self._state.speech_id + 1
        self._state = self._fresh_state(speech_id=current_speech_id)
        return self._build_incremental_output(translation)

    def set_source_language(self, language: str) -> None:
        self.source_lang = language

    def set_target_language(self, language: str) -> None:
        self.target_lang = language
        self.target_sep = "" if language in ["Chinese", "Japanese"] else " "

    def tokens_to_string(self, tokens: List[str]) -> str:
        if self.latency_unit in ["word", "spm"]:
            return " ".join(tokens)
        if self.latency_unit == "char":
            return "".join(tokens)
        raise NotImplementedError(f"Unsupported latency_unit: {self.latency_unit}")

    def clear(self) -> None:
        self._state = self._fresh_state(speech_id=self._state.speech_id)
