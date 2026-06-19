"""SimAlign word alignment (segfreetk-style) with XLM-RoBERTa on CPU."""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoConfig, AutoModel, AutoTokenizer

from streaming_sfm import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

try:
    import networkx as nx
    from networkx.algorithms.bipartite.matrix import from_biadjacency_matrix
except ImportError:  # pragma: no cover - optional at import time
    nx = None
    from_biadjacency_matrix = None

XLMR_MODEL_NAME = "xlm-roberta-base"


def _load_embedding_model(model_name: str, device: torch.device) -> torch.nn.Module:
    """Load XLM-R on CPU with int8 quantization to keep GPU memory free."""
    config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)

    if device.type != "cpu":
        model = AutoModel.from_pretrained(model_name, config=config)
        model.eval()
        model.to(device)
        return model

    try:
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig
        from transformers import TorchAoConfig

        quant_config = Int8DynamicActivationInt8WeightConfig()
        quantization_config = TorchAoConfig(quant_type=quant_config)
        model = AutoModel.from_pretrained(
            model_name,
            config=config,
            device_map=device,
            quantization_config=quantization_config,
        )
        logger.info("Loaded %s with torchao int8 quantization on CPU", model_name)
    except ImportError:
        model = AutoModel.from_pretrained(model_name, config=config)
        model.eval()
        model.to(device)
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        logger.info(
            "torchao unavailable; loaded %s with torch quantize_dynamic on CPU",
            model_name,
        )

    model.eval()
    return model


class EmbeddingLoader:
    def __init__(
        self,
        model: str = XLMR_MODEL_NAME,
        device: torch.device = torch.device("cpu"),
        layer: int = 8,
    ):
        self.model = model
        self.device = device
        self.layer = layer
        self.emb_model = _load_embedding_model(model, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        logger.info(
            "Initialized EmbeddingLoader with model=%s device=%s layer=%s",
            self.model,
            self.device,
            self.layer,
        )

    def prewarm(self, sizes: Tuple[int, ...] = (2, 4, 8, 16, 32)) -> None:
        start = time.perf_counter()
        for size in sizes:
            warmup = ("a " * size).split()
            self.get_embed_list([warmup, warmup])
        logger.info("Word aligner prewarm finished in %.2fs", time.perf_counter() - start)

    def get_embed_list(self, sent_batch: List[List[str]]) -> torch.Tensor:
        with torch.no_grad():
            if not isinstance(sent_batch[0], str):
                inputs = self.tokenizer(
                    sent_batch,
                    is_split_into_words=True,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
            else:
                inputs = self.tokenizer(
                    sent_batch,
                    is_split_into_words=False,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
            hidden = self.emb_model(**inputs.to(device=self.device))["hidden_states"]
            if self.layer >= len(hidden):
                raise ValueError(
                    f"Layer {self.layer} is out of range for model with {len(hidden)} layers."
                )
            outputs = hidden[self.layer]
            return outputs[:, 1:-1, :]


class SentenceAligner:
    def __init__(
        self,
        model: str = "xlmr",
        token_type: str = "bpe",
        distortion: float = 0.0,
        matching_methods: str = "mai",
        device: str = "cpu",
        layer: int = 8,
    ):
        model_names = {
            "bert": "bert-base-multilingual-cased",
            "xlmr": XLMR_MODEL_NAME,
        }
        all_matching_methods = {
            "a": "inter",
            "m": "mwmf",
            "i": "itermax",
            "f": "fwd",
            "r": "rev",
        }

        if model in model_names:
            model = model_names[model]
        self.model = model
        self.token_type = token_type
        self.distortion = distortion
        self.matching_methods = [all_matching_methods[m] for m in matching_methods]
        self.device = torch.device(device)
        self.embed_loader = EmbeddingLoader(model=self.model, device=self.device, layer=layer)

    @staticmethod
    def get_max_weight_match(sim: np.ndarray) -> np.ndarray:
        if nx is None or from_biadjacency_matrix is None:
            raise ValueError("networkx is required for the mwmf matching method.")

        def permute(edge):
            if edge[0] < sim.shape[0]:
                return edge[0], edge[1] - sim.shape[0]
            return edge[1], edge[0] - sim.shape[0]

        graph = from_biadjacency_matrix(csr_matrix(sim))
        matching = nx.max_weight_matching(graph, maxcardinality=True)
        matching = [permute(edge) for edge in matching]
        matching = sorted(matching, key=lambda x: x[0])
        result = np.zeros_like(sim)
        for i, j in matching:
            result[i, j] = 1
        return result

    @staticmethod
    def get_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (cosine_similarity(x, y) + 1.0) / 2.0

    @staticmethod
    def average_embeds_over_words(
        bpe_vectors: np.ndarray, word_tokens_pair: List[List[str]]
    ) -> List[np.ndarray]:
        word_to_bpe = []
        count = 0
        word_to_bpe.append([])
        for word_list in word_tokens_pair[0]:
            word_to_bpe[0].append([])
            for _ in word_list:
                word_to_bpe[0][-1].append(count)
                count += 1
        count = 0
        word_to_bpe.append([])
        for word_list in word_tokens_pair[1]:
            word_to_bpe[1].append([])
            for _ in word_list:
                word_to_bpe[1][-1].append(count)
                count += 1

        new_vectors = []
        for lang_id in range(2):
            word_vectors = []
            for word_set in word_to_bpe[lang_id]:
                word_vectors.append(bpe_vectors[lang_id][word_set].mean(0))
            new_vectors.append(np.array(word_vectors))
        return new_vectors

    @staticmethod
    def get_alignment_matrix(sim_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _, n = sim_matrix.shape
        forward = np.eye(n)[sim_matrix.argmax(axis=1)]
        backward = np.eye(sim_matrix.shape[0])[sim_matrix.argmax(axis=0)]
        return forward, backward.transpose()

    @staticmethod
    def apply_distortion(sim_matrix: np.ndarray, ratio: float = 0.5) -> np.ndarray:
        shape = sim_matrix.shape
        if shape[0] < 2 or shape[1] < 2 or ratio == 0.0:
            return sim_matrix

        pos_x = np.array(
            [[y / float(shape[1] - 1) for y in range(shape[1])] for _ in range(shape[0])]
        )
        pos_y = np.array(
            [[x / float(shape[0] - 1) for x in range(shape[0])] for _ in range(shape[1])]
        )
        distortion_mask = 1.0 - ((pos_x - np.transpose(pos_y)) ** 2) * ratio
        return np.multiply(sim_matrix, distortion_mask)

    @staticmethod
    def iter_max(sim_matrix: np.ndarray, max_count: int = 2) -> np.ndarray:
        alpha_ratio = 0.9
        m, n = sim_matrix.shape
        forward = np.eye(n)[sim_matrix.argmax(axis=1)]
        backward = np.eye(m)[sim_matrix.argmax(axis=0)]
        inter = forward * backward.transpose()

        if min(m, n) <= 2:
            return inter

        count = 1
        while count < max_count:
            mask_x = 1.0 - np.tile(inter.sum(1)[:, np.newaxis], (1, n)).clip(0.0, 1.0)
            mask_y = 1.0 - np.tile(inter.sum(0)[np.newaxis, :], (m, 1)).clip(0.0, 1.0)
            mask = ((alpha_ratio * mask_x) + (alpha_ratio * mask_y)).clip(0.0, 1.0)
            mask_zeros = 1.0 - ((1.0 - mask_x) * (1.0 - mask_y))
            if mask_x.sum() < 1.0 or mask_y.sum() < 1.0:
                mask *= 0.0
                mask_zeros *= 0.0

            new_sim = sim_matrix * mask
            fwd = np.eye(n)[new_sim.argmax(axis=1)] * mask_zeros
            bac = np.eye(m)[new_sim.argmax(axis=0)].transpose() * mask_zeros
            new_inter = fwd * bac

            if np.array_equal(inter + new_inter, inter):
                break
            inter = inter + new_inter
            count += 1
        return inter

    def get_word_aligns(
        self, src_sent: Union[str, List[str]], trg_sent: Union[str, List[str]]
    ) -> Dict[str, List[Tuple[int, int]]]:
        if isinstance(src_sent, str):
            src_sent = src_sent.split()
        if isinstance(trg_sent, str):
            trg_sent = trg_sent.split()

        l1_tokens = [self.embed_loader.tokenizer.tokenize(word) for word in src_sent]
        l2_tokens = [self.embed_loader.tokenizer.tokenize(word) for word in trg_sent]
        bpe_lists = [[bpe for w in sent for bpe in w] for sent in [l1_tokens, l2_tokens]]

        if self.token_type == "bpe":
            l1_b2w_map = []
            for i, word_list in enumerate(l1_tokens):
                l1_b2w_map.extend([i for _ in word_list])
            l2_b2w_map = []
            for i, word_list in enumerate(l2_tokens):
                l2_b2w_map.extend([i for _ in word_list])

        vectors = self.embed_loader.get_embed_list([src_sent, trg_sent]).cpu().detach().numpy()
        vectors = [vectors[i, : len(bpe_lists[i])] for i in (0, 1)]

        if self.token_type == "word":
            vectors = self.average_embeds_over_words(vectors, [l1_tokens, l2_tokens])

        all_mats: Dict[str, np.ndarray] = {}
        sim = self.get_similarity(vectors[0], vectors[1])
        sim = self.apply_distortion(sim, self.distortion)
        all_mats["fwd"], all_mats["rev"] = self.get_alignment_matrix(sim)
        all_mats["inter"] = all_mats["fwd"] * all_mats["rev"]
        if "mwmf" in self.matching_methods:
            all_mats["mwmf"] = self.get_max_weight_match(sim)
        if "itermax" in self.matching_methods:
            all_mats["itermax"] = self.iter_max(sim)

        aligns = {method: set() for method in self.matching_methods}
        for i in range(len(vectors[0])):
            for j in range(len(vectors[1])):
                for method in self.matching_methods:
                    if all_mats[method][i, j] > 0:
                        if self.token_type == "bpe":
                            aligns[method].add((l1_b2w_map[i], l2_b2w_map[j]))
                        else:
                            aligns[method].add((i, j))
        return {method: sorted(pairs) for method, pairs in aligns.items()}
