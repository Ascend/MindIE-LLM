# Expert Parallel

MoE models support expert parallelism (EP), which deploys experts on different devices to implement expert-level parallel computing.

Currently, two EP forms are implemented:

1. EP based on AllGather communication (`"ep\_level": 1`)

2. EP based on AllToAll and communication-computing fusion (`"ep\_level": 2`)

## Constraints

- The DeepSeek-V2, DeepSeek-V3, and DeepSeek-R1 models support this feature.
- If the number of parallel experts exceeds 32, DeepSeek-V3 and DeepSeek-R1 automatically enable the grouped matmul fused operator to improve computing performance.

## Parameter Description

[Table 1](#table1) describes the serving parameters required for enabling the Expert Parallel feature.

**Table 1** Expert Parallel parameters in  `models` of `ModelConfig` <a id="table1"></a>

|Parameter|Value Type|Value Range|Configuration Description|
|--|--|--|--|
|deepseekv2|-|-|-|
|ep_level|int|[1,2]|EP implementation form.<br>`1`: EP based on AllGather communication<br>`2`: EP based on AllToAll and communication-computing fusion<br>If `ep_level` is set to `2` when two servers are deployed, the two servers must be connected through a switch. Otherwise, the service will fail to be started.|
|enable_init_routing_cutoff|bool|<ul><li>true</li><li>false</li></ul>|Whether to allow topk truncation.<br>The default value is `false` (disabling the feature).<br>This parameter can be set when `ep_level` is set to `1`.|
|topk_scaling_factor|float|(0,1]|Topk truncation parameter.<br>When `ep_level` is set to `1`, the latter part of `hidden_states` on each device is invalid data. You can set the truncation parameter to reduce the graphics memory overhead.<br>In this case, you also need to set `enable_init_routing_cutoff` to `true`.|
|alltoall_ep_buffer_scale_factors|list[list[int, float]]|Each member in the list contains two numbers. The first number is a non-negative integer, and the second number is a floating-point number greater than 0. The members are sorted in descending order based on the first number.|Size of the AllToAll communication buffer. The second-level list contains two elements. The first number is the sequence length, and the second number is the buffer coefficient. The sequence length is the condition for determining the buffer coefficient. Example:<br>[[1048576, 1.32], [524288, 1.4], [262144, 1.53], [131072, 1.8], [32768, 3.0], [8192, 5.2], [0, 8.0]]<br>You are advised to configure this parameter when `ep_level` is set to `2` and you need to manage the graphics memory in a refined manner.<br>This parameter does not take effect when `ep_level` is set to `1`.|

## Usage Examples

Example when `ep_level` is set to `2`:

```json
"ModelDeployConfig" :
{
   "maxSeqLen" : 2560,
   "maxInputTokenLen" : 2048,
   "truncation" : 0,
   "ModelConfig" : [
     {
         "modelInstanceType" : "Standard",
         "modelName" : "DeepSeek-R1_w8a8",
         "modelWeightPath" : "/data/weights/DeepSeek-R1_w8a8",
         "worldSize" : 8,
         "cpuMemSize" : 5,
         "npuMemSize" : -1,
         "backendType" : "atb",
         "trustRemoteCode" : false,
         "moe_ep": 8,
         "models": {
             "deepseekv2": {
                 "ep_level": 2,
                 "alltoall_ep_buffer_scale_factors": [[1048576, 1.32], [524288, 1.4], [262144, 1.53], [131072, 1.8], [32768, 3.0], [8192, 5.2], [0, 8.0]]
             }
         }
      }
   ]
}
```

> [!NOTE]NOTE
> Generally, you are not advised to add `"alltoall_ep_buffer_scale_factors"`.

Example when `ep_level` is set to `1` in the long sequence scenario:

```json
"ModelDeployConfig" :
{
   "maxSeqLen" : 66000,
   "maxInputTokenLen" : 65000,
   "truncation" : 0,
   "ModelConfig" : [
     {
         "modelInstanceType" : "Standard",
         "modelName" : "DeepSeek-R1_w8a8",
         "modelWeightPath" : "/data/weights/DeepSeek-R1_w8a8",
         "worldSize" : 8,
         "cpuMemSize" : 5,
         "npuMemSize" : -1,
         "backendType" : "atb",
         "trustRemoteCode" : false,
         "moe_ep": 8,
         "models": {
             "deepseekv2": {
                 "ep_level": 1,
                 "enable_init_routing_cutoff": true,
                 "topk_scaling_factor": 0.25
             }
         }
      }
   ]
}
```

## Inference <a name="section1271638122016"></a>

1. Set serving parameters. This feature must be used together with MindIE Motor. Add the corresponding parameters to the serving `config.json` file based on [Parameter Description](#parameter-description). For details about the serving parameters, see [Configuration Parameters (Serving)](../user_manual/service_parameter_configuration.md).
2. Start the service. For details, see "Quick Start" \> "[Starting the Service](https://gitcode.com/Ascend/MindIE-Motor/blob/v3.0.0/docs/en/user_guide/quick_start.md)" in *MindIE Motor Developer Guide*.
