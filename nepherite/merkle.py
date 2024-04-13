from typing import Any

from nepherite.utils import sha256


class MerkleTree:
    class Node:
        hash: bytes
        left: Any
        right: Any

    def __init__(self, items: list[bytes]) -> None:
        self.items = items
        self.root = self.build_tree(0, len(items) - 1)

    def build_tree(self, l: int, r: int) -> Node:  # noqa: E741
        if l == r:
            return MerkleTree.Node(sha256(self.items[l]), None, None)

        m = (l + r) >> 1
        left = self.build_tree(l, m)
        right = self.build_tree(m + 1, r)
        return MerkleTree.Node(sha256(left.hash + right.hash), left, right)

    def get_verify_data(self, index: int) -> list[bytes]:  # noqa: A002
        res = []
        l = 0  # noqa: E741
        r = len(self.items) - 1
        while l < r:
            m = (l + r) >> 1
            if index <= m:
                res.append(self.items[m + 1])
                r = m
            else:
                res.append(self.items[l])
                l = m + 1  # noqa: E741
        return res
