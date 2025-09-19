import torch
import torch.nn as nn
    
class R_Embedding(nn.Module):
    """
    径向嵌入：
    输入 r_ij: [B,N,N,1] 或 [B,N,N]
    输出 R: [B,N,N,out_dim]
    """
    def __init__(self, r_c, num_basis, hidden_dims, out_dim, p_env=6, eps=1e-8):
        super().__init__()
        self.r_c = float(r_c)
        self.num_basis = int(num_basis)
        self.p_env = int(p_env)
        self.eps = float(eps)
        # b 注册为 buffer（不可训练），可以随 model.to(device) 移动
        b = torch.arange(1, self.num_basis + 1, dtype=torch.get_default_dtype())
        self.register_buffer('b', b)  # shape [num_basis]
        # MLP
        layers = []
        in_dim = self.num_basis
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.SiLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def f_env(self, d):
        # d = r / r_c
        p = self.p_env
        x = d
        # 多项式包络（NequIP 风格）
        term = 1 - (p+1)*(p+2)/2 * x**p + p*(p+2) * x**(p+1) - p*(p+1)/2 * x**(p+2)
        env = term * (x < 1.0).to(x.dtype)
        return env

    def B(self, r_ij):
        # r_ij: [...,1] or [...]
        if r_ij.dim() == 3:
            # [B,N,N] -> add last dim
            r_ij = r_ij.unsqueeze(-1)
        # d: [...,1]
        d = (r_ij / (self.r_c + self.eps)).clamp(min=0.0)
        r_safe = r_ij.clamp(min=self.eps)
        # b: [num_basis] -> reshape以便广播为 [..., num_basis]
        lead = r_ij.shape[:-1]
        b = self.b.view(*([1]*len(lead)), -1)  # broadcastable
        sin_term = torch.sin(b * torch.pi * d)  # [..., num_basis]
        out = 2.0 / (self.r_c + self.eps) * (sin_term / r_safe) * self.f_env(d)
        # 返回 [..., num_basis]
        return out.squeeze(-2) if out.shape[-1]==1 else out

    def forward(self, r_ij):
        # 支持 [B,N,N,1] 或 [B,N,N]
        if r_ij.dim() == 3:
            r_ij = r_ij.unsqueeze(-1)
        B_shape = r_ij.shape[:-1]
        B_flat = self.B(r_ij).reshape(-1, self.num_basis)  # [B*N*N, num_basis]
        R_flat = self.mlp(B_flat)  # [B*N*N, out_dim]
        out = R_flat.reshape(*B_shape, -1)  # [B,N,N,out_dim]
        return out
    
    

if __name__ == "__main__":
    # 测试R_Embedding
    r_ij = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)  # 示例距离
    r_c = 5.0  
    num_basis = 16  # 基嵌入维度
    hidden_dims = [64, 32]  # MLP隐藏层维度
    out_dim = 1  

    r_embedding = R_Embedding(r_c, num_basis, hidden_dims, out_dim)
    output = r_embedding(r_ij)
    print(output)  # 输出结果