# MoE Communication Mechanism Details
<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-05-28T08:46:15.375Z pushedAt=2026-05-29T03:18:57.492Z -->

## 1. Why Does MoE Require Communication?

The core idea of the Mixture of Experts (MoE) model is to distribute model capacity across multiple "Expert" sub-networks. During training or inference, not all experts are activated; instead, different tokens are assigned to different experts for processing based on the decisions of the gating network (Gate).

Due to memory limitations, we often cannot place all experts on a single card, thus requiring **Expert Parallelism**. This introduces the need for communication.

1. **Token dispatch**: Tokens on the current card may be dispatched to experts on other cards. Therefore, the token data needs to be sent from the current card to the card holding the target expert.

2. **Expert computation**: After receiving the tokens, the target device performs computations using its local expert network.

3. **Combination**: After computation, the results must be sent back to the original card holding the token to facilitate subsequent layer computations or loss aggregation.

This process involves extensive cross-device data exchange, so communication efficiency directly determines the training and inference speed of MoE models.

## 2. Detailed Explanation of Core Communication Methods

The implementation of MoE primarily relies on the following communication methods:

### 2.1 AllGather

* **Function**: Each participating device broadcasts its own data shard to all other devices, so that every card ultimately obtains a complete concatenated copy of all devices' data.

* **Application in MoE**:

  * **Dispatch**: Sends tokens to all devices based on routing information.

  * **Combination**: Sends the results after expert computation back to all devices.

* **Characteristics**: Simple to implement with clear semantics, but the data volume grows linearly with the number of cards, making it suitable for small data broadcasts.

### 2.2 AllToAll

* **Function**: Each card splits the data it holds and sends different data blocks to different target cards, while simultaneously receiving data blocks from different cards.

* **Application in MoE**:

  * **Dispatch**: Sends Tokens to the cards holding the corresponding experts based on routing information.

  * **Combination**: Sends the results after expert computation back to the original card that owns the Token.

* **Characteristics**: The complex communication patterns and uneven data distribution (imbalanced load) make it the bottleneck of MoE communication.

### 2.3 Merged Compute and Communication (MC2)

* **Background**: Standard AllToAll uses a **synchronous blocking** mode, during which the compute units are idle, and it lacks specialized optimization for sparse routing.

* **Functionality**: Achieves **parallel overlap** of communication and expert computation through **asynchronous communication** and a **sparse mask mechanism**.

* **Features**:

    **Sparse Communication**: Transmits only active tokens based on **mc2_mask**.
    **Compute-Communication Overlap**: Use the paired operators **npu_moe_distribute_dispatch(_v2)** and **npu_moe_distribute_combine(_v2)** to return immediately after dispatch without blocking. While waiting for communication, computation proceeds in parallel, achieving overlap between communication and computation.

### 2.4 Fused Merged Compute and Communication (FusedMC2)

* **Function**: Merges **dispatch**, **ffn**, and **combine** into a single large fused operator.

* **Advantages**:

  * Significantly reduces Kernel Launch overhead.

  * Improves memory bandwidth utilization.

  * Hides communication latency (Overlap).

## 3. Communication Method Selection

`device_type`: Hardware model, `910B`/`910_93` are Ascend NPU models, `any` means unlimited.

`is_prefill`: computation stage, `prefill` or `decode`.

`world_size`: The number of parallel cards, used to determine if large-scale parallel optimization conditions are met (e.g., ≥16).

`quant_type`: The quantization type, where `W4A8_DYNAMIC` is 4-bit weight quantization, `other` indicates other or no quantization, and `any` means unlimited.

`ep_size`: The degree of Expert Parallelism, where `≤32` indicates small-scale expert parallelism, and `any` means unlimited.

`tokens vs cap`: The comparison between the number of input tokens and the maximum capacity of the MC2 operator (must be ≤ capacity).

`moe_tp`: MoE tensor parallelism switch. `√` = enabled, `x` = disabled.

`attn_dp`: Attention data parallelism switch. `√` = enabled, `x` = disabled.

| Index | device_type | is_prefill | world_size | quant_type      | ep_size | tokens vs cap | moe_tp | attn_dp | Selected Strategy |
|:-----:|:-----------:|:----------:|:----------:|:---------------:|:-------:|:-------------:|:------:|:-------:|:-----------------:|
| 1     | 910B        | decode     | ≥16        | any             | any     | any           | x      | x       | MC2               |
| 2     | 910B        | prefill    | ≥16        | other           | any     | any           | x      | √       | ALLTOALL          |
| 3     | 910B        | decode     | ≥16        | any             | any     | any           | √      | x       | ALLGATHER         |
| 4     | 910B        | decode     | <16        | W4A8_DYNAMIC    | any     | any           | x      | x       | ALLTOALL          |
| 5     | 910B        | decode     | <16        | other           | any     | any           | x      | x       | ALLGATHER         |
| 6     | 910B        | prefill    | any        | W4A8_DYNAMIC    | any     | any           | x      | x       | ALLTOALL          |
| 7     | 910B        | prefill    | any        | other           | any     | any           | x      | x       | ALLGATHER         |
| 8     | 910_93      | prefill    | any        | any             | ≤32     | ≤cap          | x      | x       | FUSED_MC2         |
| 9     | 910_93      | prefill    | any        | any             | >32     | any           | x      | x       | ALLTOALL          |
| 10    | 910_93      | decode     | any        | any             | ≤32     | ≤cap          | x      | x       | FUSED_MC2         |
| 11    | 910_93      | decode     | any        | any             | ≤32     | >cap          | x      | x       | MC2               |
| 12    | 910_93      | decode     | any        | any             | any     | >cap          | x      | x       | Error, unsupported |
| 13    | any         | any        | any        | any             | any     | any           | x      | √       | ALLTOALL          |
| 14    | any         | any        | any        | any             | any     | any           | √      | √       | Error, unsupported |
