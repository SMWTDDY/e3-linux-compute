#NequIPModel
import torch
import torch.nn as nn
from EquivariantConv import EquivariantConvLayer
from find_neighbor import build_adjacent_pairs_torch,rebuild_by_indices

r_c = 5.0
# Interaction Block（卷积 + per-l 自作用）和完整模型堆叠（3 层）
class InteractionBlock(nn.Module):
    def __init__(self, l_max=3, channels_per_l=None,num_basis=None, **conv_kwargs):
        super().__init__()
        self.conv = EquivariantConvLayer(l_max=l_max, channels_per_l=channels_per_l,num_basis=num_basis, r_c=r_c,**conv_kwargs)
        # per-l 的 self interaction（1x1）
        self.self_linear = nn.ModuleDict()
        self.non = nn.SiLU()
        for l in range(l_max+1):
            C = channels_per_l[l]
            # 先把 (C,2*l+1) 展平到 (C*(2*l+1)) 做全连接，然后再 reshape 回 (C,2*l+1)
            self.self_linear[str(l)] = nn.Linear(C * (2*l+1), C * (2*l+1))
    def forward(self, feats, pos, neighbor_mask=None):
        conv_out = self.conv(feats, pos, neighbor_mask)
        out = {}
        for l in conv_out:
            B,N,C,M = conv_out[l].shape
            x = conv_out[l].reshape(B,N,C*M)
            x = self.self_linear[str(l)](x)
            x = self.non(x)
            out[l] = x.reshape(B,N,C,M)
        return out
    
class NequIPModel(nn.Module):
    def __init__(self, l_max=3, channels_per_l=None, num_basis=16 ,num_layers=3, max_z=100, conv_kwargs=None):
        super().__init__()

        self.l_max = l_max
        self.channels_per_l = channels_per_l
        self.embedding = nn.Embedding(max_z, channels_per_l[0])  # z -> l0 channels
        conv_kwargs = conv_kwargs or {}
        self.blocks = nn.ModuleList([InteractionBlock(l_max=l_max, channels_per_l=channels_per_l, num_basis=num_basis,**conv_kwargs) \
            for _ in range(num_layers)])
        self.out_lin = nn.Linear(channels_per_l[0], 1)  # l=0 channels -> 原子标量能量
        self.act = nn.SiLU()  
        
    def forward(self, z, pos, neighbor_mask):
        """
        z: [B,N] int 原子序数
        pos: [B,N,3]
        neighbor_mask: [3,E] bool 或 None
        返回: total_energy [B], atomic_energy [B,N], feats dict
        """
        B,N = z.shape
        device = pos.device
        if not pos.requires_grad:
            pos = pos.requires_grad_(True)
        emb = self.embedding(z)  # [B,N,C0]
        feats = {0: emb.unsqueeze(-1)}  # [B,N,C0,1]
        for l in range(1, self.l_max+1):
            C = self.channels_per_l[l]
            feats[l] = torch.zeros(B,N,C,2*l+1, device=device, dtype=emb.dtype)
        # 堆叠交互块
        for blk in self.blocks:
            out = blk(feats, pos, neighbor_mask)            
            new_feats = {}
            for l in range(self.l_max+1):
                
                # shape检查和断言
                if feats[l].shape == out[l].shape:
                    new_feats[l] = feats[l] + out[l]
                else:
                    print(f"Shape mismatch at l={l}: feats[{l}].shape={feats[l].shape}, out[{l}].shape={out[l].shape}")
                    B, N = z.shape
                    expected_shape = (B, N, self.channels_per_l[l], 2*l+1)
                    assert out[l].shape == expected_shape, f"out[{l}].shape={out[l].shape}, expected={expected_shape}"
                    new_feats[l] = out[l]
                    
            new_feats = {l: self.act(new_feats[l]) for l in new_feats}  # 使用 SiLU 激活函数
            feats = new_feats
        atomic_scalars = feats[0].squeeze(-1)  # [B,N,C0]
        atomic_energy = self.out_lin(atomic_scalars).squeeze(-1)  # [B,N]
        E = atomic_energy.sum(dim=1)
        F = -torch.autograd.grad(outputs=E, inputs=pos, grad_outputs= torch.ones_like(E, device=E.device), create_graph=self.training)[0]
        return E, F




if __name__ == '__main__':
    # 假设: l_max=2, 每个l的通道数为[8, 4, 2]，3层
    l_max = 2
    # 修改为 dict 格式
    channels_per_l = {0: 8, 1: 4, 2: 2}
    num_layers = 3
    model = NequIPModel(l_max=l_max, channels_per_l=channels_per_l, num_layers=num_layers)
    

    B, N = 2, 3
    z = torch.randint(1, 10, (B, N))           # 原子序数，范围1-9
    pos = torch.randn(B, N, 3)          
    pos2 =torch.randn(B,N,3)# 原子坐标
    print(pos)
    # 假设每个分子有 E 条边（这里以全连接为例，E = N*N）
    E = N * N
    neighbor_mask = build_adjacent_pairs_torch(pos,r_c)
    neighbor_mask2 = build_adjacent_pairs_torch(pos2,r_c)
    # 前向传播
    total_energy ,F= model(z, pos, neighbor_mask)
    total_energy2 ,F2= model(z, pos2, neighbor_mask2)
 
    print("Total energy shape:", total_energy.shape)  # [B]
    print("forces shape:", F.shape)  # [B,N,3]
    print("Total energy diff:", total_energy-total_energy2)
    print("Forces:", F)
    print("Forces2:", F2)