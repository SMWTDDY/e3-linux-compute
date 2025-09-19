#Clebsch_Gordan coefficients
import numpy as np
from math import factorial
def clebsch(j1, j2, j3, m1, m2, m3):
    """Calculates the Clebsch-Gordon coefficient
    for coupling (j1,m1) and (j2,m2) to give (j3,m3).

    Parameters
    ----------
    j1 : float
        Total angular momentum 1.

    j2 : float
        Total angular momentum 2.

    j3 : float
        Total angular momentum 3.

    m1 : float
        z-component of angular momentum 1.

    m2 : float
        z-component of angular momentum 2.

    m3 : float
        z-component of angular momentum 3.

    Returns
    -------
    cg_coeff : float
        Requested Clebsch-Gordan coefficient.

    """
    if m3 != m1 + m2:
        return 0
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0
    
    vmin = int(np.max([-j1 + j2 + m3, -j1 + m1, 0]))
    vmax = int(np.min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))

    C = np.sqrt((2.0 * j3 + 1.0) * factorial(int(j3 + j1 - j2)) *
                factorial(int(j3 - j1 + j2)) * factorial(int(j1 + j2 - j3) )*
                factorial(int(j3 + m3)) * factorial(int(j3 - m3)) /
                (factorial(int(j1 + j2 + j3 + 1))) *
                factorial(int(j1 - m1)) * factorial(int(j1 + m1)) *
                factorial(int(j2 - m2)) * factorial(int(j2 + m2)))
    S = 0
    for v in range(vmin, vmax + 1):
        S += (-1.0) ** (v + j2 + m2) / factorial(int(v) )* \
            factorial(int(j2 + j3 + m1 - v) )* factorial(int(j1 - m1 + v)) / \
            factorial(int(j3 - j1 + j2 - v)) / factorial(int(j3 + m3 - v))/ \
            factorial(int(v + j1 - j2 - m3))
    C = C * S
    return C


def build_cg_table(l1, l2, l3, clebsch_func=clebsch):
    """
    生成 CG 系数的 3D 数组：shape = (2*l1+1, 2*l2+1, 2*l3+1)
    m 索引顺序为 -l..+l
    """
    m1s = list(range(-l1, l1+1))
    m2s = list(range(-l2, l2+1))
    m3s = list(range(-l3, l3+1))
    arr = np.zeros((2*l1+1, 2*l2+1, 2*l3+1), dtype=np.float64)
    for i,m1 in enumerate(m1s):
        for j,m2 in enumerate(m2s):
            for k,m3 in enumerate(m3s):
                if m1 + m2 != m3:
                    arr[i,j,k] = 0.0
                else:
                    arr[i,j,k] = float(clebsch_func(float(l1), float(l2), float(l3), float(m1), float(m2), float(m3)))
    return arr


if __name__ == "__main__":
    # Example usage
    j1, j2, j3 = 1.0, 1.0, 2.0
    m1, m2, m3 = 1.0, -1.0, 0.0
    cg_coeff = clebsch(j1, j2, j3, m1, m2, m3)
    print(f"Clebsch-Gordan coefficient for ({j1},{m1}) and ({j2},{m2}) to ({j3},{m3}): {cg_coeff}")
    
    cg_table = build_cg_table(1, 1, 2)
    print("Clebsch-Gordan table for (1, 1, 2):")
    print(cg_table)
    
    
    