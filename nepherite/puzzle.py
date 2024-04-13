from typing import TypeVar

Answer = TypeVar("Answer")

class Puzzle:
    @staticmethod
    def compute(data: bytes) -> Answer:
        pass
    
    @staticmethod
    def verify(data: bytes, answer: Answer) -> bool:
        pass
