# todo: blockConentCLass, hash types, transactionCoentn type
'''
This is the request message for blocks.
Only  to request a block, not to return it.
'''
class PullBlockMessage:
    parentId: int
    blockHash: int

'''
This is the request message for blocks.
'''
class SendBlockMessage:
    parentId: int
    blockHash: int 
    blockContent: int

'''
Only to request transactions
'''
class PullTransactionMessage:
    parentId: int
    transactionHash: int

'''
Only to send the transaction
'''
class PullTransactionMessage:
    parentId: int
    transactionHash: int
    transactionContent: int