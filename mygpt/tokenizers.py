from abc import abstractmethod
from typing import Self, override


class Tokenizer:
    @classmethod
    @abstractmethod
    def from_data(cls, content: str) -> Self: ...

    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, data: list[int]) -> str: ...

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...


class CharTokenizer(Tokenizer):
    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self._stoi = {t: i for i, t in enumerate(tokens)}

    @override
    @classmethod
    def from_data(cls, content: str) -> Self:
        tokens = sorted(set(content))
        return cls(tokens)

    @override
    def encode(self, text: str) -> list[int]:
        return [self._stoi[ch] for ch in text]

    @override
    def decode(self, data: list[int]) -> str:
        return "".join(self._tokens[i] for i in data)

    @property
    @override
    def vocab_size(self) -> int:
        return len(self._tokens)


class BigramTokenizer(Tokenizer):
    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self._stoi = {t: i for i, t in enumerate(tokens)}

    @override
    @classmethod
    def from_data(cls, content: str) -> Self:
        chars = sorted(set(content))
        tokens = [a + b for a in chars for b in chars]
        return cls(tokens)

    @override
    def encode(self, text: str) -> list[int]:
        if len(text) % 2 != 0:
            text += " "  # padding
        return [self._stoi[text[i : i + 2]] for i in range(0, len(text), 2)]

    @override
    def decode(self, data: list[int]) -> str:
        return "".join(self._tokens[i] for i in data)

    @property
    @override
    def vocab_size(self) -> int:
        return len(self._tokens)
