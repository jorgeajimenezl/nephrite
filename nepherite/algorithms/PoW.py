'''
Idea:
    given the transaction hashes as Th1,..., Thn
    and the block hash Bh, the PoW will consist on
    find a solution to a linear equation system described as:
    
    y = Ax

    with y = Bh, A = [W  ..] , W = [Th1, ..., Thn]
                     [.. ..]
        
    x any vector that is solution to the equation, A a matrix with W as submatriz 
    but only at the beggining of it.


    easy to check , only multiply x and A that retuns y, and that A containg
    W in the upper left part.
'''

import numpy as np

def linear_equation_check(y: np.ndarray, A: np.ndarray, x: np.ndarray) -> bool:
    pass