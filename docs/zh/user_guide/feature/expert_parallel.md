# Expert Parallel

MoE类模型支持Expert Parallel（EP，专家并行），通过将专家分别部署在不同的设备上，实现专家级别的并行计算。

当前实现两种形式的EP并行：

1. 基于AllGather通信的EP并行，即"ep\_level": 1

2. 基于AllToAll和通算融合的EP并行，即"ep\_level": 2

## 限制与约束

-   DeepSeek-V2，DeepSeek-V3，DeepSeek-R1模型支持对接此特性。
-   当专家并行数超过32时，DeepSeek-V3、DeepSeek-R1自动使能Grouped MatMul融合算子，提升计算性能。

## 参数说明

开启Expert Parallel特性，需要配置的服务化参数如[表1](#table1)所示。

**表 1**  Expert Parallel特性补充参数：**ModelConfig中的models参数** <a id="table1"></a>

|配置项|取值类型|取值范围|配置说明|
|--|--|--|--|
|deepseekv2|
|ep_level|int|[1,2]|专家并行的实现形式。<br>1：表示基于AllGather通信的EP并行<br>2：表示基于AllToAll和通算融合的EP<br>并行双机部署场景下且"ep_level"设置为“2”时，两台服务器必须通过交换机连接，否则拉起服务会失败。|
|enable_init_routing_cutoff|bool|truefalse|是否允许topk截断。<br>默认值：false（关闭）<br>“ep_level”=“1”时，可配置该参数。|
|topk_scaling_factor|float|(0,1]|topk截断参数。<br>“ep_level”=“1”时，每台设备的hidden_states后段部分为无效数据，可设置截断参数减小显存开销。<br>需同时配置“enable_init_routing_cutoff”=“true”。|
|alltoall_ep_buffer_scale_factors|list[list[int, float]]|列表每个成员包含两个数：第一个数为非负整数，第二个数为大于0的浮点数。排列顺序按照第一个数的大小降序排列。|AllToAll通信buffer大小，第二层list包含两个元素，第一个数为序列长度、第二个数为buffer系数。序列长度为buffer系数的选择判断条件。示例：<br>[[1048576, 1.32], [524288, 1.4], [262144, 1.53], [131072, 1.8], [32768, 3.0], [8192, 5.2], [0, 8.0]]<br>“ep_level”=“2”时，且用户需要精细化地管理显存的时候建议配置该项。<br>“ep_level”=“1”时该参数配置不生效。|


## 使用样例

“ep\_level”=“2”时使用样例：

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
},
```

> [!NOTE]说明
> 一般情况下不建议添加"alltoall\_ep\_buffer\_scale\_factors"。

“ep\_level”=“1”时长序列场景使用样例：

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
},
```

## 执行推理<a name="section1271638122016"></a>

1. 配置服务化参数。该特性需配合MindIE Motor使用，按照[参数说明](#参数说明)在服务化的config.json文件中添加相应参数。服务化参数说明请参见[配置参数说明（服务化）](../user_manual/service_parameter_configuration.md)章节。
2. 启动服务。具体请参考《MindIE Motor开发指南》中的“快速入门 \> [启动服务](https://gitcode.com/Ascend/MindIE-Motor/blob/dev/docs/zh/user_guide/quick_start.md)”章节。

