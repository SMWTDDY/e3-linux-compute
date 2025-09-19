import torch
import math

def factorial(n: int, device=None, dtype=torch.float32):
    n_t = torch.tensor(float(n) + 1.0, device=device, dtype=dtype)
    return torch.exp(torch.lgamma(n_t))

def legendre(l, x):
    if l == 0:
        return torch.ones_like(x)
    elif l == 1:
        return x
    else:
        return ((2 * l - 1) * x * legendre(l - 1, x) - (l - 1) * legendre(l - 2, x)) / l

#计算连带勒让德多项式
def assoc_legendre(l, m, x):
    # 使用递推方法计算关联勒让德多项式 P_l^m(x)（m >= 0），随后根据 m 的符号处理
    m_int = int(abs(m))
    if m_int > l:
        return torch.zeros_like(x)
    # P_m^m(x) = (-1)^m (2m-1)!! (1 - x^2)^{m/2}
    if m_int == 0:
        P_mm = torch.ones_like(x)
    else:
        df = 1.0
        for k in range(1, m_int + 1):
            df *= (2 * k - 1)
        P_mm = ((-1.0) ** m_int) * df * (1.0 - x * x) ** (m_int / 2)
    if l == m_int:
        P_lm = P_mm
    else:
        # P_{m+1}^m(x) = x (2m+1) P_m^m(x)
        P_m1m = x * (2 * m_int + 1) * P_mm
        if l == m_int + 1:
            P_lm = P_m1m
        else:
            P_prev = P_mm
            P_curr = P_m1m
            for ll in range(m_int + 2, l + 1):
                P_next = ((2 * ll - 1) * x * P_curr - (ll - 1 + m_int) * P_prev) / (ll - m_int)
                P_prev, P_curr = P_curr, P_next
            P_lm = P_curr
    # 如果 m 为负，使用关系 P_l^{-m} = (-1)^m * (l-m)!/(l+m)! * P_l^{m}
    if m < 0:
        sign = (-1.0) ** m_int
        ratio = (factorial(l - m_int) / factorial(l + m_int))
        return sign * ratio * P_lm
    return P_lm

#球谐函数
def Y(l, m, theta, phi):
    # theta, phi: torch张量
    x = torch.cos(theta)
    m_abs = abs(m)
    Plm = assoc_legendre(l, m_abs, x)
    norm = torch.sqrt((2 * l + 1) / (4 * math.pi) * factorial(l - m_abs) / factorial(l + m_abs))
    if m == 0:
        return (norm * Plm).real if hasattr(norm * Plm, 'real') else norm * Plm
    elif m > 0:
        return torch.sqrt(torch.tensor(2.0, device=theta.device)) * norm * Plm * torch.cos(m * phi)
    else:
        return torch.sqrt(torch.tensor(2.0, device=theta.device)) * norm * Plm * torch.sin(m_abs * phi)




def Y_l(l, r):
    r = torch.as_tensor(r, dtype=torch.float32)
    r0 = r
    orig_shape = list(r0.shape)
    # 展平成 [B*N*N, 3]
    r_flat = r.view(-1, 3)
    # 计算范数并保护除零
    eps = 1e-8
    r_norm = torch.norm(r_flat, dim=-1, keepdim=True)
    zero_mask = (r_norm.squeeze(-1) <= eps)
    r_unit = r_flat / (r_norm + eps)
    x, y, z = r_unit[..., 0], r_unit[..., 1], r_unit[..., 2]  #[B*N*N,3]->[B*N*N]
    # 如果 x 和 y 同时为 0，则加上 eps
    eps = 1e-8
    zero_xy_mask = (x == 0) & (y == 0)
    x = x + zero_xy_mask * eps
    y = y + zero_xy_mask * eps
    # 计算 theta 和 phi
    theta = torch.acos(torch.clamp(z, -1.0, 1.0))
    phi = torch.atan2(y, x)

        
    V_list = []
    for m in range(-l, l + 1):
        V_list.append(Y(l, m, theta, phi))
    V = torch.stack(V_list, dim=-1)  # [B*N*N, 2l+1]
    out_shape = orig_shape[:-1] + [2 * l + 1]
    V = V.view(*out_shape)
    return V

if __name__ =="__main__":
    #测试球谐函数
    B,N = 2,3
    r = torch.rand(B,N,3)
    a = r.unsqueeze(1) - r.unsqueeze(2)  # [B,N,N,3]
    Ylm = Y_l(2, a)
    print(Ylm)  # [2,2,2,5]
    
    

    