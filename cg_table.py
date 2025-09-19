import numpy as np
import math
from fractions import Fraction


def clebsch_scalar(j1, j2, j3, m1, m2, m3):
    if m1 + m2 != m3:
        return 0.0
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0.0
    # 转为 int 用于 factorial
    j1i, j2i, j3i = int(round(j1)), int(round(j2)), int(round(j3))
    m1i, m2i, m3i = int(round(m1)), int(round(m2)), int(round(m3))
    from math import factorial, sqrt

    def f(n: int) -> int:
        assert n == round(n)
        return factorial(round(n))

    try:
        C = (
            (2.0 * j3 + 1.0)
            * Fraction(
                f(j3 + j1 - j2) * f(j3 - j1 + j2) * f(j1 + j2 - j3) * f(j3 + m3i) * f(j3 - m3i),
                f(j1 + j2 + j3 + 1) * f(j1 - m1i) * f(j1 + m1i) * f(j2 - m2i) * f(j2 + m2i),
            )
        ) ** 0.5
    except Exception:
        return 0.0

    # 求和 S
    vmin = int(max([-j1i + j2i + m3i, -j1i + m1i, 0]))
    vmax = int(min([j2i + j3i + m1i, j3i - j1i + j2i, j3i + m3i]))

    S = 0
    for v in range(vmin, vmax + 1):
        denom = (f(v) * f(j3i - j1i + j2i - v) * f(j3i + m3i - v) * f(v + j1i - j2i - m3i))
        if denom == 0:
            continue
        term = (-1) ** int(v + j2i + m2i) * Fraction(
            f(j2i + j3i + m1i - v) * f(j1i - m1i + v), denom
        )
        S += term
    return float(C * S)


def build_cg_table(l1, l2, l3, clebsch_func=clebsch_scalar):
    """
    生成 CG 系数的 3D 数组：shape = (2*l1+1, 2*l2+1, 2*l3+1)
    m 索引顺序为 -l..+l
    """
    m1s = list(range(-l1, l1 + 1))
    m2s = list(range(-l2, l2 + 1))
    m3s = list(range(-l3, l3 + 1))
    arr = np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1), dtype=np.float64)
    for i, m1 in enumerate(m1s):
        for j, m2 in enumerate(m2s):
            for k, m3 in enumerate(m3s):
                if m1 + m2 != m3:
                    arr[i, j, k] = 0.0
                else:
                    arr[i, j, k] = float(
                        clebsch_func(float(l1), float(l2), float(l3), float(m1), float(m2), float(m3))
                    )
    return arr



if __name__ =="__main__":
    # 测试 CG 系数计算
    l1, l2, l3 = 1, 1, 2
    cg_table = build_cg_table(l1, l2, l3)
    print("CG coefficients for (l1=1, l2=1, l3=2):")
    print(cg_table) 