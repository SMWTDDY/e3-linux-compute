import torch
def compute_relative_coords(pos):
    """
    计算批量分子中每个原子相对于其他原子的坐标
    
    参数:
        pos: 原子坐标张量，形状 [batch_size, atom_num, 3]    
    返回:
        relative_coords: 相对坐标张量，形状 [batch_size, atom_num, atom_num, 3]
                         relative_coords[b, i, j] 表示第b个分子中，原子j相对于原子i的坐标
    """
    # 扩展维度以支持广播：
    # pos_ij 形状: [batch_size, atom_num, 1, 3]（为每个原子i扩展一个维度用于广播）
    # pos_ji 形状: [batch_size, 1, atom_num, 3]（为每个原子j扩展一个维度用于广播）
    pos_ij = pos.unsqueeze(2)  # 在第2维插入维度
    pos_ji = pos.unsqueeze(1)  # 在第1维插入维度
    
    # 广播计算所有原子对的相对坐标：j - i
    relative_coords = pos_ji - pos_ij 
    
    return relative_coords

# 测试示例
if __name__ == "__main__":
    # 构造输入：2个分子，每个分子3个原子，3D坐标
    batch_size = 2
    atom_num = 3
    pos = torch.tensor([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # 第1个分子
        [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]   # 第2个分子
    ])
    
    # 计算相对坐标
    rel_coords = compute_relative_coords(pos)
    
    print("输入形状:", pos.shape)  # 输出: torch.Size([2, 3, 3])
    print("输出形状:", rel_coords.shape)  # 输出: torch.Size([2, 3, 3, 3])
    
    # 打印第1个分子中，原子1相对于原子0的坐标（预期: [1.0, 0.0, 0.0]）
    print("\n第1个分子中原子1相对于原子0的坐标:", rel_coords[0, 0, 1])
    print("mask:",rel_coords)
