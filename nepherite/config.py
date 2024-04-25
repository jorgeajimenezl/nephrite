import os

BLOCK_SIZE = os.getenv("BLOCK_SIZE", 4)
BLOCK_REWARD = os.getenv("BLOCK_REWARD", 100)
RETRY_COUNT = os.getenv("RETRY_COUNT", 3)
BLOCK_TTL = os.getenv("BLOCK_TTL", 1)
CHAIN_GAP_SIZE = os.getenv("CHAIN_GAP_SIZE", 1)
SHIBA_RATE = os.getenv("SHIBA_RATE", 2)
