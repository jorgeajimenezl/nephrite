from cryptography.hazmat.primitives import hashes


def sha256(data: bytes) -> bytes:
    digest = hashes.Hash(hashes.SHA3_256())
    digest.update(data)
    return digest.finalize()