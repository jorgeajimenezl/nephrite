# Consensus Algorithm: PoW
## Topology:
Nodes need a minimum of 8 and a maximum of 125 connections to be part of the network.
In other words, nodes will seek connections within the network until reaching 125 or a lesser number chosen.

## Gossip:
- Push: Messages are sent to all connected nodes.
- Pull: Information is requested from a node.
    - Pull involves making a push with the request for what is needed.
    - Then, nodes that do not have that information store who sent the pull request and broadcast the pull message.
    - Traditional parent style.
Messages are repeated only 3 times.

## Transactions:
- Push: When they are created.
- Pull: When needed?

## Blocks:
- Push: To the header when created.
- Pull: To the Full block when requested.

## Types:

### Messages:
- PullTransaction
- SendTransaction
- PullBlock
- SendBlock-Header
- SendFullBlock
- Ping

### Transactions:


### Blocks:


### Implement PoW implies challenge and validation 
ChallengePoW(contentOfTheBlock)=> answer Slow
ValidatePoW(contentOfTheBlock, answer) => bool Fast
answer = nonce


-
-