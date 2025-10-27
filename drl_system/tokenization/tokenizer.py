"""Simple modular tokenizer for experimentation."""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List

from ..config import TokenizerConfig


class DynamicTokenizer:
    """A minimal tokenizer that can grow over time via vocabulary updates."""

    def __init__(self, config: TokenizerConfig) -> None:
        self.config = config
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._init_special_tokens()

    def _init_special_tokens(self) -> None:
        for token in [
            self.config.pad_token,
            self.config.unk_token,
            self.config.bos_token,
            self.config.eos_token,
        ]:
            self._add_token(token)

    def _add_token(self, token: str) -> None:
        if token in self.token_to_id:
            return
        index = len(self.token_to_id)
        if index >= self.config.vocab_size:
            raise ValueError("Vocabulary capacity exceeded")
        self.token_to_id[token] = index
        self.id_to_token[index] = token

    def fit(self, corpus: Iterable[str], min_freq: int = 2) -> None:
        counter: Counter[str] = Counter()
        for text in corpus:
            counter.update(text.split())
        for token, freq in counter.most_common():
            if freq < min_freq:
                continue
            if token not in self.token_to_id and len(self.token_to_id) < self.config.vocab_size:
                self._add_token(token)

    def encode(self, text: str) -> List[int]:
        tokens = [self.config.bos_token] + text.split() + [self.config.eos_token]
        return [self.token_to_id.get(tok, self.token_to_id[self.config.unk_token]) for tok in tokens]

    def decode(self, token_ids: Iterable[int]) -> str:
        tokens = [self.id_to_token.get(i, self.config.unk_token) for i in token_ids]
        without_special = [
            tok
            for tok in tokens
            if tok not in {self.config.bos_token, self.config.eos_token, self.config.pad_token}
        ]
        return " ".join(without_special)

    def grow(self, new_tokens: Iterable[str]) -> None:
        for tok in new_tokens:
            if len(self.token_to_id) >= self.config.vocab_size:
                break
            self._add_token(tok)


__all__ = ["DynamicTokenizer"]
