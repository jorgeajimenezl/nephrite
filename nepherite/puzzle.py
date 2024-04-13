from typing import Generic, TypeVar

Answer = TypeVar("Answer")

class Puzzle(Generic[Answer]):
    @staticmethod
    def compute(data: bytes) -> Answer:
        pass
    
    @staticmethod
    def verify(data: bytes, answer: Answer) -> bool:
        pass
