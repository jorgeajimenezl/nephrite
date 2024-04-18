from unittest import TestCase

from nepherite.merkle import MerkleTree
from nepherite.utils import sha256


class MerkleTreeTests(TestCase):
    def test_build_tree(self):
        """
        Test that the Merkle tree is being built correctly.
        """
        items = [b"hello", b"world", b"foo", b"bar"]
        merkle_tree = MerkleTree(items)

        # Test leaf nodes
        self.assertEqual(merkle_tree.root.left.left.hash, b"hello")
        self.assertEqual(merkle_tree.root.left.right.hash, b"world")
        self.assertEqual(merkle_tree.root.right.left.hash, b"foo")
        self.assertEqual(merkle_tree.root.right.right.hash, b"bar")

        # Test node hash computation
        left_hash = sha256(b"hello" + b"world")
        right_hash = sha256(b"foo" + b"bar")
        self.assertEqual(merkle_tree.root.left.hash, left_hash)
        self.assertEqual(merkle_tree.root.right.hash, right_hash)
        self.assertEqual(merkle_tree.root.hash, sha256(left_hash + right_hash))
