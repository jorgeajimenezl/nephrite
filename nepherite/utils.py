import logging

from cryptography.hazmat.primitives import hashes

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def sha256(*items: list[bytes]) -> bytes:
    digest = hashes.Hash(hashes.SHA3_256())
    for item in items:
        digest.update(item)
    return digest.finalize()
