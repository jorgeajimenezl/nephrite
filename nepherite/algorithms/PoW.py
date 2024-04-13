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
    try:
        can_pass = len(A.shape) == 2 and \
                len(y.shape) == 1 and \
                len(x.shape) == 1 and \
                A.shape[0] == y.shape[0] and \
                A.shape[1] == x.shape[0]

        if not can_pass:
            return False

        y_hat = A @ x
        return np.array_equal(y, y_hat)
    except:
        return False