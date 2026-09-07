"""Thin dictionary/callable bridges; no framework is a core dependency."""

from .base import PolicyAdapter, TraceDecoder, TrainerAdapter
from .datasets import (
    from_fisher_alignments,
    from_seamless_pair,
    from_seamless_transcripts,
    resolve_seamless_roles,
)

__all__ = [
    "PolicyAdapter",
    "TraceDecoder",
    "TrainerAdapter",
    "from_fisher_alignments",
    "from_seamless_pair",
    "from_seamless_transcripts",
    "resolve_seamless_roles",
]
