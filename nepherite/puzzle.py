from typing import Tuple
import hashlib

Answer = Tuple[int, str]
DIFFICULTY = 5

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
        nonce, hash_value = answer
        prefix = '0' * DIFFICULTY
        data_app = data + nonce.to_bytes(4, byteorder='big')  # Use 4 bytes for nonce
        computed_hash = hashlib.sha256(data_app).hexdigest()
        return computed_hash.startswith(prefix)
