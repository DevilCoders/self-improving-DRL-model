import pytest

pytest.importorskip("numpy")

from drl_system.config import TokenizerConfig
from drl_system.tokenization.tokenizer import DynamicTokenizer


def test_tokenizer_roundtrip():
    config = TokenizerConfig(vocab_size=32)
    tokenizer = DynamicTokenizer(config)
    tokenizer.fit(["hello world", "hello agent"], min_freq=1)
    encoded = tokenizer.encode("hello world")
    decoded = tokenizer.decode(encoded)
    assert "hello" in decoded and "world" in decoded
