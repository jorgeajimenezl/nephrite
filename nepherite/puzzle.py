from typing import Generic, TypeVar

from nepherite.utils import sha256

DIFFICULTY = 5
Answer = TypeVar("Answer")

class Puzzle(Generic[Answer]):
    @staticmethod
    def compute(data: bytes) -> Answer:
        pass
    
    @staticmethod
    def verify(data: bytes, answer: Answer) -> bool:
        pass

class HashNoncePuzzle(Puzzle[int]):
    @staticmethod
    def compute(data: bytes) -> int:
        nonce = 0
        while True:
            if HashNoncePuzzle.verify(data, nonce):
                return nonce
            nonce += 1
    
    @staticmethod
    def verify(data: bytes, answer: int) -> bool:
        hash = sha256(data, answer.to_bytes(4))  # noqa: A001
        return all(hash[i] == 0 for i in range(DIFFICULTY))
