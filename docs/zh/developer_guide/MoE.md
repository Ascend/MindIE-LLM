# MoE 通信机制详解

## 1. 为什么 MoE 需要通信？

混合专家模型（Mixture of Experts, MoE）的核心思想是将模型容量分散到多个“专家”（Expert）子网络中。在训练或推理时，并非所有专家都会被激活，而是根据门控路由网络（Gate）的决策，将不同的 Token 分配给不同的专家处理。

由于显存限制，我们通常无法将所有专家放在同一张显卡上，因此需要使用 **专家并行（Expert Parallelism）**。这就导致了通信需求：

1. **Token 路由（Dispatch）**：当前显卡上的 Token 可能被路由到其他显卡上的专家。因此，需要将 Token 数据从当前显卡发送到持有目标专家的显卡。
2. **专家计算（Expert Computation）**：目标显卡接收 Token 后，利用本地专家网络进行计算。
3. **结果合并（Combine）**：计算完成后，需要将结果发送回原始持有该 Token 的显卡，以便进行后续层的计算或损失汇总。

这个过程涉及大量的跨卡数据交换，因此通信效率直接决定了 MoE 模型的训练和推理速度。

## 2. 核心通信方式详解

在 MoE 的实现中，主要依赖以下几种通信方式：

### 2.1 AllGather

* **功能**：每个参与通信的设备将自己持有的数据分片广播给所有其他设备，最终每张卡都获得所有设备数据的完整拼接副本。
* **在 MoE 中的应用**：
    * **Dispatch 阶段**：将 Token 根据路由信息发送到所有设备上。
    * **Combine 阶段**：将专家计算后的结果发送回所有设备上。
* **特点**：实现简单，语义清晰，但数据量随卡数线性增长，适合小数据广播。

### 2.2 AllToAll

* **功能**：每张显卡把持有数据切分后将不同的数据块发送给不同的目标显卡，同时接收来自不同显卡的数据块。
* **在 MoE 中的应用**：
    * **Dispatch 阶段**：将 Token 根据路由信息发送到持有对应专家的显卡。
    * **Combine 阶段**：将专家计算后的结果发送回 Token 所属的原始显卡。
* **特点**：通信模式复杂，数据分布不均匀（负载不均衡），是 MoE 通信的瓶颈所在。

### 2.3 MC2 (Merged Compute and Communication)

* **背景**：标准 AllToAll 采用**同步阻塞**模式，通信期间计算单元空闲，且缺乏对稀疏路由的专门优化。
* **功能**：通过**异步通信**与**稀疏掩码机制**，实现通信与专家计算的**并行重叠**。
* **特点**：
    **稀疏通信**：基于 **mc2_mask** 仅传输活跃 Token。
    **通算重叠**：使用 **npu_moe_distribute_dispatch(_v2)** 、**npu_moe_distribute_combine(_v2)**配套算子，dispatch后立刻返回不阻塞，在等待通信时并行计算，实现通算掩盖。

### 2.4 FusedMC2 (Fused Merged Compute and Communication)

* **功能**：将 **dispatch** 、 **ffn** 与 **combine** 合并为一个大融合算子。
* **优势**：
    * 显著降低 Kernel Launch 开销。
    * 提高显存带宽利用率。
    * 隐藏通信延迟（Overlap）。

## 3. 通信方式选择

**device_type 含义**：硬件型号，`910B`/`910_93` 为昇腾 NPU 型号，`any` 表示不限制。

**is_prefill 含义**：计算阶段，`prefill` 为预填充阶段，`decode` 为解码生成阶段。

**world_size 含义**：并行卡数，用于判断是否满足大规模并行优化条件（如 ≥16）。

**quant_type 含义**：量化类型，`W4A8_DYNAMIC` 为 4bit 权重量化，`other` 为其他或无量化，`any` 表示不限制。

**ep_size 含义**：专家并行度，`≤32` 为小规模专家并行，`any` 表示不限制。

**tokens vs cap 含义**：输入 token 数与 MC2 算子最大容量的对比关系（需 ≤ capacity）。

**moe_tp 含义**：MoE 张量并行开关，`√` 启用，`x` 禁用。

**attn_dp 含义**：Attention 数据并行开关，`√` 启用，`x` 禁用。

| 序号 | device_type | is_prefill | world_size | quant_type      | ep_size | tokens vs cap | moe_tp | attn_dp | 选中策略      |
|:----:|:-----------:|:----------:|:----------:|:---------------:|:-------:|:-------------:|:------:|:-------:|:-------------:|
| 1    | 910B        | decode     | ≥16        | any             | any     | any           | x      | x       | MC2           |
| 2    | 910B        | prefill    | ≥16        | other           | any     | any           | x      | √       | ALLTOALL      |
| 3    | 910B        | decode     | ≥16        | any             | any     | any           | √      | x       | ALLGATHER     |
| 4    | 910B        | decode     | <16        | W4A8_DYNAMIC    | any     | any           | x      | x       | ALLTOALL      |
| 5    | 910B        | decode     | <16        | other           | any     | any           | x      | x       | ALLGATHER     |
| 6    | 910B        | prefill    | any        | W4A8_DYNAMIC    | any     | any           | x      | x       | ALLTOALL      |
| 7    | 910B        | prefill    | any        | other           | any     | any           | x      | x       | ALLGATHER     |
| 8    | 910_93      | prefill    | any        | any             | ≤32     | ≤cap          | x      | x       | FUSED_MC2     |
| 9    | 910_93      | prefill    | any        | any             | >32     | any           | x      | x       | ALLTOALL      |
| 10   | 910_93      | decode     | any        | any             | ≤32     | ≤cap          | x      | x       | FUSED_MC2     |
| 11   | 910_93      | decode     | any        | any             | ≤32     | >cap          | x      | x       | MC2           |
| 12   | 910_93      | decode     | any        | any             | any     | >cap          | x      | x       | 报错，不支持   |
| 13   | any         | any        | any        | any             | any     | any           | x      | √       | ALLTOALL      |
| 14   | any         | any        | any        | any             | any     | any           | √      | √       | 报错，不支持   |
