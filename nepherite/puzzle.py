from typing import Generic, TypeVar

DIFFICULTY = 5
Answer = TypeVar("Answer")

class Puzzle(Generic[Answer]):
    @staticmethod
    def compute(data: bytes) -> Answer:
        pass
    
    @staticmethod
    def verify(data: bytes, answer: Answer) -> bool:
        pass

class HashPuzzle(Puzzle[bytes]):
    @staticmethod
    def compute(data: bytes) -> bytes:
        nonce = 0
        prefix = '0' * DIFFICULTY
        while True:
            data_app = data + nonce.to_bytes(4, byteorder='big')
            hash_value = hashlib.sha256(data_app).hexdigest()
            if hash_value.startswith(prefix):
                return nonce, hash_value
            nonce += 1
    
    @staticmethod
    def verify(data: bytes, answer: bytes) -> bool:
        nonce, hash_value = answer
        prefix = '0' * DIFFICULTY
        data_app = data + nonce.to_bytes(4, byteorder='big')  # Use 4 bytes for nonce
        computed_hash = hashlib.sha256(data_app).hexdigest()
        return computed_hash.startswith(prefix)
