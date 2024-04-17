from ipv8.test.base import TestBase

from nepherite.base import Blockchain


class BlockchainCommunityTests(TestBase):
    async def test_start_communities(self):
        """
        Test Blockchain Community started correctly.
        """
        self.initialize(Blockchain, 2)
