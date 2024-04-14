from typing import Tuple
import hashlib

from nepherite.utils import sha256

DIFFICULTY = 5
Answer = TypeVar("Answer")

class Puzzle:
    @staticmethod
    def compute(data: bytes) -> Answer:
        nonce = 0
        prefix = '0' * DIFFICULTY
        while True:
            data_app = data + nonce.to_bytes(4, byteorder='big')  # Use 4 bytes for nonce
            
            hash_value = hashlib.sha256(data_app).hexdigest()
            
            if hash_value.startswith(prefix):
                return nonce, hash_value
            
            nonce += 1
    
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
