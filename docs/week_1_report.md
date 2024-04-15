# Whitepaper
## System

This is our current report of our system.

### Consensus Algorithm

We have implemented the proof of work (PoW) consensus algorithm, opting for a widely used approach. In this method, the block data is converted into bytes, and the task involves finding a nonce to append at the end of the block, ensuring that the resulting hash begins with a predetermined number of zeros. For our academic purposes, we have set this number to 5.

## Topology

Nodes within the network are required to establish a minimum of 8 connections and can have a maximum of 125 connections. In practical terms, nodes will actively seek connections within the network until they have reached either the maximum limit of 125 connections or a lower number determined by the node operator.

## Pull / Push

- **Push**: The messages are sent to all connected nodes.
- **Pull**: To find and receive data from other nodes, we opt for a traditional parent structure for the pull messages. That is, each pull message will contain the sender (parent) to trace back the route to the original sender.
- Messages will only be repeated 3 times.

## Block Structure

![Block Structure](/imgs/drawing.png)

## Tests

## Limitations

## Progress
### Week 1

- [x] Consensus puzzle
- [x] Verify Transaction Functionality
- [x] Verify Block Functionality
- [x] Test of 100 peers
- [x] Functional Transaction
- [ ] Functional Verified Chain of Minimun 3 Blocks
- [x] Local Database per peer
- [x] Week 1 report