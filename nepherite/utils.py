from cryptography.hazmat.primitives import hashes


def sha256(*items: list[bytes]) -> bytes:
    digest = hashes.Hash(hashes.SHA3_256())
    for item in items:
        digest.update(item)
    return digest.finalize()
