import torch
#pos:[B,N,3]
def build_adjacent_pairs_torch(pos, r_c):
    dists = torch.cdist(pos,pos) #[B,N,N]
    #构造bool矩阵
    mask = (dists < r_c) & (~torch.eye(pos.shape[1], dtype=torch.bool, device=pos.device))\
        .unsqueeze(0).repeat(pos.shape[0],1,1)
    pairs = mask.nonzero(as_tuple=False)
    edge_index = pairs.t().contiguous()    # shape [3,E]
    return edge_index

#按照得到索引重构原bool张量
def rebuild_by_indices(pos,edge_index):
    b_indices = edge_index[0]
    x_indices = edge_index[1]
    y_indices = edge_index[2]
    B,N,_ = pos.shape
    fal = torch.zeros(B,N,N,dtype =torch.bool,device=pos.device)
    fal[b_indices,x_indices,y_indices] = True  
    return fal
    
if __name__ == "__main__":
    pos = torch.rand(4 , 2 , 3)  # 2批，1个原子，每个原子3D坐标
    print("Coordinates:", pos)  # 打印坐标
    r_c = 20.5
    edge_index = build_adjacent_pairs_torch(pos, r_c)
    print("Edge index shape:", edge_index.shape)  
    print("Edge index:", edge_index)  # 打印边索引
    b_indices = edge_index[0]
    x_indices = edge_index[1]
    y_indices = edge_index[2]
    #还原  [3,E]->[B,N,N](bool)
    #3*E = B*N*N - num_False
    B,N,_ = pos.shape
    fal = torch.zeros(B,N,N,dtype =torch.bool,device=pos.device)
    fal[b_indices,x_indices,y_indices] = True  
    print(fal)
    rebuild = rebuild_by_indices(pos,edge_index)
    print("rebuild=",rebuild)
