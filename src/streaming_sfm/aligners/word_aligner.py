"""Word-level aligner wrapper around SimAlign."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from streaming_sfm import LOG_LEVEL
from streaming_sfm.aligners.simalign import SentenceAligner

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

_ALIGNMENT_METHODS = ("inter", "mwmf", "itermax", "fwd", "rev")


@dataclass
class AlignmentResult:
    source_tokens: list[str]
    target_tokens: list[str]
    alignments: list[tuple[int, int]]
    alignment_method: str


class WordAligner:
    def __init__(
        self,
        device: str = "cpu",
        alignment_model: str = "xlmr",
        matching_methods: str = "a",
        prewarm: bool = True,
    ):
        logger.info(
            "Loading SimAlign with model=%s on device=%s",
            alignment_model,
            device,
        )
        self.mt_aligner = SentenceAligner(
            device=device,
            model=alignment_model,
            token_type="bpe",
            matching_methods=matching_methods,
        )
        if prewarm:
            self.mt_aligner.embed_loader.prewarm()

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [tok for tok in text.strip().split() if tok]

    def align(self, src_tokens: list[str], tgt_tokens: list[str]) -> AlignmentResult:
        if not src_tokens or not tgt_tokens:
            return AlignmentResult(
                source_tokens=src_tokens,
                target_tokens=tgt_tokens,
                alignments=[],
                alignment_method="empty",
            )

        try:
            alignments_dict = self.mt_aligner.get_word_aligns(src_tokens, tgt_tokens)
        except Exception:
            logger.exception(
                "Word alignment failed for src=%r tgt=%r",
                " ".join(src_tokens),
                " ".join(tgt_tokens),
            )
            return AlignmentResult(
                source_tokens=src_tokens,
                target_tokens=tgt_tokens,
                alignments=[],
                alignment_method="failed",
            )

        raw_alignments = None
        method_used = "itermax"
        for method in _ALIGNMENT_METHODS:
            if method in alignments_dict and alignments_dict[method]:
                raw_alignments = alignments_dict[method]
                method_used = method
                break

        alignments = sorted((src_idx, tgt_idx) for src_idx, tgt_idx in (raw_alignments or []))
        return AlignmentResult(
            source_tokens=src_tokens,
            target_tokens=tgt_tokens,
            alignments=alignments,
            alignment_method=f"SimAlign-{method_used}",
        )

    def align_text(self, src_text: str, tgt_text: str) -> AlignmentResult:
        return self.align(self.tokenize(src_text), self.tokenize(tgt_text))
