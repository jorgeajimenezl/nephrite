import hashlib
from typing import Tuple

# find nonce, such as hash(meessage + nonce ) start with dificulty 0s

# difuculty 5: 1.25 seconds
# difuculty 6: 11.07 seconds
# difuculty 7: 38.09 seconds

def solve_puzzle(message: str, difficulty: int) -> Tuple[int, str]:
    nonce = 0
    prefix = '0' * difficulty
    while True:
        # Concatenate the message with the nonce
        data = message + str(nonce)
        
        hash_value = hashlib.sha256(data.encode()).hexdigest()
        
        # Check if the hash value meets the difficulty requirement
        if hash_value.startswith(prefix):
            return nonce, hash_value
        
        nonce += 1