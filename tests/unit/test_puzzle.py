from unittest import TestCase

from nepherite.puzzle import DIFFICULTY, HashNoncePuzzle
from nepherite.utils import sha256


class PuzzleTests(TestCase):
    def testPuzzle(self):
        """
        Test that the Puzzle class is working correctly.
        """
        data = b"hello"
        answer = HashNoncePuzzle.compute(data)
        self.assertTrue(HashNoncePuzzle.verify(data, answer))

        # Test verification
        data_hash = sha256(data, answer.to_bytes(4))
        self.assertTrue(all(data_hash[i] == 0 for i in range(DIFFICULTY)))
