# Performance Tuning

## Tuning on the CPU

### Enabling the CPU high-performance mode

``` bash
cpupower -c all frequency-set -g performance
```

### Enabling the transparent huge page

``` bash
echo always > /sys/kernel/mm/transparent_hugepage/enabled
```

### Enabling jemalloc tuning

To optimize jemalloc, you need to compile the jemalloc dynamic link library and import the compiled dynamic link library to the script. The procedure is as follows:

1. Download the [jemalloc source code](https://github.com/jemalloc/jemalloc) and compile and install it by referring to the `INSTALL.md` file.
2. Before starting the service, import the jemalloc dynamic link library to the environment by running the following command:

``` bash
export LD_PRELOAD=${path_to_lib}/libjemalloc.so:$LD_PRELOAD
```

`${path_to_lib}` indicates the path of the `libjemalloc.so`.

## Scheduling Features

### Asynchronous Scheduling

The MindIE inference process is executed synchronously. An inference process can be divided into the following three phases:

- Data preparation phase (executed on the CPU)
- Model inference phase (executed on the NPU)
- Data return phase (executed on the CPU)

Asynchronous scheduling leverages the time consumed during the model inference phase to mask the time taken in the data preparation and return phases. Specifically, it utilizes NPU computation time to mask CPU-side operations, excluding sampling-related overhead. However, requests carrying the EOS flag (inference termination) are repeatedly processed, resulting in unnecessary consumption of NPU computing and graphics memory resources. This feature is suitable for scenarios involving a large `maxBatchSize` and long input/output sequences.

Set the environment variable to enable the asynchronous scheduling feature.

``` bash
export MINDIE_ASYNC_SCHEDULING_ENABLE=1
```

### PD Disaggregation

PD Disaggregation refers to the practice of instantiating and deploying the prefill and decode stages of model inference on different machine resources for concurrent inference. Given that prefill is compute-intensive and decode is memory-intensive, this approach adjusts the ratio of P/D nodes to increase the batch size on decode nodes, fully utilizing NPU compute capacity and improving overall cluster throughput.
In decode scenarios that require low latency, prefill-decode disaggregation can better leverage performance advantages compared to prefill-decode co-location.

For details, see [PD Disaggregation](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindieservice/servicedev/mindie_service0049.html).

## Parallel Parameters

### Tensor Parallelism (TP)

TP distributes tensors (e.g., weight matrices and activations) across multiple devices (e.g., NPUs) for distributed inference.
By default, all models use TP with a partition size equal to `worldSize`.

Beyond standard tensor parallelism, models such as DeepSeek-V3, DeepSeek-R1, and DeepSeek-V3.1 support local TP partition for the Lmhead matrix and the O projection matrix. This feature is recommended for PD disaggregation where D nodes are not distributed, as it reduces matrix computation time and lowers inference latency.

| Configuration Item           | Value| Value Range                | Configuration Description                                                   |
| ---------------- | ------- | ---------------------- | ---------------------------------------------------------- |
| tp               | int     | [1, worldSize]         | Number of tensor parallel processes.                                                 |
| lm_head_local_tp | int     | [1, world size/Number of nodes]| Number of partitions for Lmhead parallelism. This parameter can be enabled only when `tp` is set to `1`. Otherwise, the value is the same as that of `tp` by default.     |
| o_proj_local_tp  | int     | [1, world size/Number of nodes]| Partition count for the Attention O matrix. This parameter can be enabled only when `tp` is set to `1`. Otherwise, the value is the same as that of `tp` by default.|

The following is an example of serving configuration for enabling local TP partition for the Lmhead matrix and O project matrix on 800I A3.

``` json
{
    "ModelConfig": [
        {
            "tp": 1,
            "models": {
                "deepseekv2": {
                    "lm_head_local_tp": 16,
                    "o_proj_local_tp": 2,
                }
            }
        }
    ]
}
```

### Data Parallelism (DP)

DP splits inference requests into multiple batches and assigns each batch to a different device for parallel processing. Each device processes distinct batches independently, and the results are merged afterward.
DP can be used together with TP, but does not currently support combination with CP.

| Configuration Item| Value| Value Range        | Configuration Description                                        |
| ----- | ------- | -------------- | ----------------------------------------------- |
| dp    | int     | [1, worldSize] | Number of data parallel processes. When combined with TP, `tp * dp` must equal `worldSize`.|

The following is an example of the configuration of serving parameters.

``` json
{
    "ModelConfig": [
        {
            "worldSize": 8,
            "dp": 2,
            "tp": 4
        }
    ]
}
```

### Sequence Parallelism (SP)

SP splits the KV cache across ranks so that each rank stores a unique portion, reducing memory usage and enabling long-sequence inference.
Currently, this feature is supported only for W8A8 quantized weights of models such as DeepSeek-V3, DeepSeek-R1, and DeepSeek-V3.1.
SP can be combined with either DP or CP.

| Configuration Item| Value| Value Range  | Configuration Description                                                     |
| ----- | ------- | -------- | ------------------------------------------------------------ |
| sp    | int     | Same as TP. | Number of KV cache partitions.<br>When combined with DP or CP, `dp * sp` or `cp * sp` must equal `worldSize`.|

The following is an example of the configuration of serving parameters.

``` json
{
    "ModelConfig": [
        {
            "worldSize": 16,
            "dp": 2,
            "tp": 8,
            "sp": 8
        }
    ]
}
```

### Context Parallelism (CP)

CP parallelizes the Self-attention module along the sequence dimension. It splits long sequences across devices for parallel processing, reducing the time to first token. 
Currently, this feature is only supported with W8A8 quantized weights for models such as DeepSeek-V3, DeepSeek-R1, and DeepSeek-V3.1. 
CP must be used together with SP, and cannot be combined with DP.

| Configuration Item| Value| Value Range| Configuration Description                                            |
| ----- | ------- | ------- | -------------------------------------------------- |
| cp    | int     | [1, 2]  | Currently, when the CP feature is enabled, the number of partitions is fixed to 2, and `cp * tp` must equal `worldSize`.|

The following is an example of the configuration of serving parameters.

``` json
{
    "ModelConfig": [
        {
            "worldSize": 16,
            "cp": 2,
            "tp": 8,
            "sp": 8
        }
    ]
}
```

### Expert Parallelism (EP)

MoE models support EP, which distributes experts across devices to enable expert-level parallel computation. 
Two EP modes are currently implemented:  

1. `ep_level=1`: EP based on AllGather communication.
2. `ep_level=2`: EP based on AllToAll with computation-communication fusion.

| Configuration Item   | Value| Value Range        | Configuration Description                                               |
| -------- | ------- | -------------- | ------------------------------------------------------ |
| ep_level | int     | [1, 2]         | EP implementation form.                                       |
| moe_tp   | int     | [1, worldSize] | MoE TP size defaults to the same as TP. When `ep_level=2`, `moe_tp` must be `1`.|
| moe_ep   | int     | [1, worldSize] | Number of EP partitions for the MoE part. Must satisfy `moe_ep * moe_tp == world_size`.        |

The following an example of the configuration of serving parameters for DeepSeek-V3.

``` json
{
    "ModelConfig": [
        {
            "worldSize": 16,
            "moe_tp": 1,
            "moe_ep": 16,
            "models": {
                "deepseekv2": {
                    "ep_level": 2
                }
            }
        }
    ]
}
```
