import torch

def insert_tensors_at_dim(tensor, dim, indices, values):
    """
    在tensor的指定维度上的若干个位置插入若干个张量
    
    Args:
        tensor: 原始tensor
        dim: 指定的维度
        indices: 要插入的位置索引列表/tensor（插入后的位置）
        values: 要插入的张量，形状应该与tensor在除了指定维度外的其他维度匹配
    
    Returns:
        插入后的新tensor
    """
    # 确保indices是tensor且为1维
    if not isinstance(indices, torch.Tensor):
        indices = torch.tensor(indices, dtype=torch.long)
    indices = indices.flatten()
    
    # 确保values是tensor
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values)
    
    # 获取原tensor的形状
    original_shape = list(tensor.shape)
    
    # 检查values的形状是否匹配
    expected_shape = original_shape.copy()
    expected_shape[dim] = len(indices)
    
    if list(values.shape) != expected_shape:
        raise ValueError(f"values shape {values.shape} doesn't match expected shape {expected_shape}")
    
    # 对indices进行排序，以便按顺序插入
    sorted_indices, sort_order = torch.sort(indices)
    sorted_values = values.index_select(dim, sort_order)
    
    # 创建结果tensor的形状
    result_shape = original_shape.copy()
    result_shape[dim] = original_shape[dim] + len(indices)
    
    # 创建结果tensor
    result = torch.zeros(result_shape, dtype=tensor.dtype, device=tensor.device)
    
    # 创建索引来追踪原tensor和新values的位置
    original_idx = 0
    insert_idx = 0
    
    for i in range(result_shape[dim]):
        if insert_idx < len(sorted_indices) and i == sorted_indices[insert_idx]:
            # 在当前位置插入新值
            result_slice = [slice(None)] * result.ndim
            result_slice[dim] = i
            
            values_slice = [slice(None)] * sorted_values.ndim
            values_slice[dim] = insert_idx
            
            result[tuple(result_slice)] = sorted_values[tuple(values_slice)]
            insert_idx += 1
        else:
            # 复制原tensor的值
            if original_idx < original_shape[dim]:
                result_slice = [slice(None)] * result.ndim
                result_slice[dim] = i
                
                original_slice = [slice(None)] * tensor.ndim
                original_slice[dim] = original_idx
                
                result[tuple(result_slice)] = tensor[tuple(original_slice)]
                original_idx += 1
    
    return result

# 使用示例
if __name__ == "__main__":
    # 示例1: 在2D tensor的第1维插入
    tensor = torch.tensor([[1, 2, 3],
                          [4, 5, 6]])  # shape: (2, 3)
    
    # 在位置1和3插入新行
    indices = [1, 3]
    values = torch.tensor([[10, 20, 30],
                          [40, 50, 60]])  # shape: (2, 3)
    
    result = insert_tensors_at_dim(tensor, dim=0, indices=indices, values=values)
    print("原tensor:")
    print(tensor)
    print("\n插入后的tensor:")
    print(result)
    
    # 示例2: 在3D tensor的第2维插入
    tensor_3d = torch.randn(2, 4, 3)
    indices_3d = [1, 3]
    values_3d = torch.randn(2, 2, 3)  # 在第1维插入2个元素
    
    result_3d = insert_tensors_at_dim(tensor_3d, dim=1, indices=indices_3d, values=values_3d)
    print(f"\n3D示例 - 原shape: {tensor_3d.shape}, 插入后shape: {result_3d.shape}")