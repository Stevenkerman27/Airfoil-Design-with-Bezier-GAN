import numpy as np

def check_intersection(coords):
    """
    Check if the polygon self-intersects.
    coords: (N, 2) array of coordinates.
    returns: True if self-intersects, False otherwise.
    """
    N = len(coords)
    A = coords[:-1]
    B = coords[1:]
    
    def ccw(A, B, C):
        return (C[..., 1] - A[..., 1]) * (B[..., 0] - A[..., 0]) - (B[..., 1] - A[..., 1]) * (C[..., 0] - A[..., 0])
    
    A_exp = A[:, None, :]
    B_exp = B[:, None, :]
    C_exp = A[None, :, :]
    D_exp = B[None, :, :]
    
    ccw1 = ccw(A_exp, C_exp, D_exp)
    ccw2 = ccw(B_exp, C_exp, D_exp)
    ccw3 = ccw(A_exp, B_exp, C_exp)
    ccw4 = ccw(A_exp, B_exp, D_exp)
    
    intersect = ((ccw1 * ccw2) < 0) & ((ccw3 * ccw4) < 0)
    mask = np.triu(np.ones((N-1, N-1), dtype=bool), k=2)
    
    return np.any(intersect & mask)

coords = np.array([[0,0], [1,0], [1,1], [0,1], [0,0]])
print("Square:", check_intersection(coords))

coords2 = np.array([[0,0], [1,1], [1,0], [0,1], [0,0]])
print("Bowtie:", check_intersection(coords2))
