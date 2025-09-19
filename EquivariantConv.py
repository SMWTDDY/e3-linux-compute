import torch
import torch.nn as nn
from cg_table import build_cg_table
from R_embedding import R_Embedding
from Ylm_harm import Y_l
from find_neighbor import rebuild_by_indices

class EquivariantConvLayer(nn.Module):
    
    def __init__(self, l_max=3, channels_per_l: dict =None, num_basis=None, r_c=5.0, cg_cache=True):
        """
        channels_per_l: 通道数对应的dict, e.g. {0:8,1:6,2:4,3:4}代第 0 层有 8 个通道，第 1 层有 6 个通道，第 2 层和第 3 层各有 4 个通道
        num_basis: R_Embedding 输出维度
        r_c: 截断半径（用于径向基）
        cg_cache: 是否预先构建 CG 表（推荐 True）
        """
        super().__init__()
        self.l_max = l_max
        if channels_per_l is None:
            channels_per_l = {l: 8 for l in range(l_max+1)} #默认每层有 8 个通道
        self.channels_per_l = channels_per_l
        # R_Embedding：把 r -> 径向权重
        self.r_emb = R_Embedding(r_c=r_c, num_basis=16, hidden_dims=[64,32], out_dim=num_basis)
        
        # 对于每个 (l_in,l_filt,l_out) 组合，创建一个通道混合线性层(Cin->Cout)
        self.mix = nn.ModuleDict() #nn.ModuleDict以字典（key-value 形式）的方式组织和管理多个子模块
        for l_in in range(l_max+1):
            Cin = channels_per_l[l_in]
            for l_filt in range(l_max+1):
                for l_out in range(abs(l_in - l_filt), min(l_max, l_in + l_filt)+1):
                    Cout = channels_per_l[l_out]
                    key = f"{l_in}_{l_filt}_{l_out}"
                    self.mix[key] = nn.Linear(Cin, Cout, bias=False)
                    
        # 预计算 CG 表（numpy），存为 dict
        self.cg_tables = {}
        if cg_cache:
            for l1 in range(l_max+1):
                for l2 in range(l_max+1):
                    for l3 in range(abs(l1-l2), min(l_max, l1+l2)+1):
                        key = (l1,l2,l3)
                        self.cg_tables[key] = build_cg_table(l1,l2,l3)  
                        
    def forward(self, feats: dict, pos: torch.Tensor, neighbor_mask: torch.Tensor = None):
        """
        feats: dict l -> [B,N,C_l, 2*l+1]
        pos: [B,N,3]
        neighbor_mask: [3,E] bool 或 None
        返回 out_feats: dict l -> [B,N,C_l,2*l+1]
        """        
        B,N,_ = pos.shape
        device = pos.device
        dtype = next(self.parameters()).dtype
        
        # 计算对原子对的位移向量 rij 和距离 rnorm
        rij = pos.unsqueeze(2) - pos.unsqueeze(1)  # [B,N,N,3]
        rnorm = torch.norm(rij, dim=-1, keepdim=True)  # [B,N,N,1]
        # 稳定化：避免除以 0
        eps = 1e-8
        rnorm_clamped = rnorm.clamp(min=eps)  # [B,N,N,1]
        # 后续用 rnorm_clamped 作为径向输入
        R = self.r_emb(rnorm_clamped)
        radial_scalar = R.mean(dim=-1, keepdim=True)  # [B,N,N,1]

        # neighbor_mask: [3,E]->[B,N,N]
        neighbor_mask = rebuild_by_indices(pos,neighbor_mask)

        # 计算球谐 Y_l(l, rij) -> [B,N,N,2*l+1] 并且创造字典保存起来
        Y = {}
        for l in range(self.l_max+1):
            Y_l_val = Y_l(l, rij)  #应返回实值张量,不能使用复数
            Y_l_val = Y_l(l, rij)  # 应返回与 rij 相关的实值张量
            Y[l] = Y_l_val.to(device=device, dtype=dtype)
        # 径向嵌入
        #r->norm(r)->basis(norm(r))
        R = self.r_emb(rnorm_clamped.to(device=device))  # [B,N,N,num_basis]
        # 简化：把 R 降维到标量权重
        radial_scalar = R.mean(dim=-1, keepdim=True)  # [B,N,N,1]
        # 准备输出容器   l--->[B,N,C_l,2*l+1]
        out_feats = {l: torch.zeros(B, N, self.channels_per_l[l], 2*l+1, device=device, dtype=dtype) for l in range(self.l_max+1)} 
        # 遍历路径 l_in x l_filt -> l_out
        #第一层for 代表输入特征的阶数l_in
        for l_in in range(self.l_max+1):
            Vin = feats.get(l_in, None)
            if Vin is None:
                continue
            # Vin: [B,N,Cin,m1]
            # 第二层for 代表球谐函数的阶数l_filt
            for l_filt in range(self.l_max+1):
                Ylf = Y[l_filt]  # [B,N,N,m2]
                # 遍历所有可能的 l_out
                for l_out in range(abs(l_in - l_filt), min(self.l_max, l_in + l_filt)+1):
                    key_cg = (l_in, l_filt, l_out)
                
                    if key_cg not in self.cg_tables:
                        continue
                    #从CG表里得到CG系数
                    # cg: [m1,m2,m3] 
                    cg_np = self.cg_tables[key_cg]  
                    cg_t = torch.as_tensor(cg_np, device=device, dtype=dtype)  #转换为tensor张量
                    
                    # 扩展并做tensorproduct
                    # Vin: [B,N,Cin,m1] -> [B,N,1,Cin,m1] -> expand -> [B,N,N,Cin,m1]
                    Vin_exp = Vin.unsqueeze(2).expand(-1, -1, N, -1, -1)  # [B,N,N,Cin,m1]
                    # Ylf: [B,N,N,m2] -> [B,N,N,1,m2]
                    Ylf_exp = Ylf.unsqueeze(3)  # [B,N,N,1,m2]
                    # prod: [B,N,N,Cin,m1,m2] = [B,N,N,Cin,m1,1] * [B,N,N,1,1,m2] 广播机制对齐
                    prod = Vin_exp.unsqueeze(-1) * Ylf_exp.unsqueeze(-2)
                    # tensordot: m1,m2 与 cg 的前两维内积 -> 得到 m3  [B,N,N,Cin,m3]
                    msg = torch.tensordot(prod, cg_t, dims=([ -2, -1 ], [0,1]))  
                    msg = msg * radial_scalar.unsqueeze(-1).to(msg.dtype)  # [B,N,N,Cin,m3]
                    
                    # mask 聚合
                    mask = neighbor_mask.to(msg.dtype).unsqueeze(-1).unsqueeze(-1)  # [B,N,N]->tensor([B,N,N,1])
                    agg = (msg * mask).sum(dim=2)  # [B,N,N,Cin,m3]*[B,N,N,1]->[B,N,N,Cin,m3]-sum(dim=2)->[B,N,Cin,m3]             
                    B_, N_, Cin_, M3 = agg.shape #[B,N,Cin,m3]
                    agg_reshaped = agg.permute(0,1,3,2).reshape(B_*N_*M3, Cin_)  #置换线性层后改变形状为[B*N*M3, Cin]，
                    mix_layer = self.mix[f"{l_in}_{l_filt}_{l_out}"] #self.mix是前面定义过的 nn.ModuleDict
                    mixed = mix_layer(agg_reshaped)  # [B*N*M3, Cout] Cout是通道数
                    Cout = self.channels_per_l[l_out]  #从通道数字典中获取每层的通道数
                    mixed = mixed.reshape(B_, N_, M3, Cout).permute(0,1,3,2)  # [B,N,Cout,m3]
                    # 累加到 out_feats[l_out]
                    out_feats[l_out] = out_feats[l_out] + mixed
        return out_feats
