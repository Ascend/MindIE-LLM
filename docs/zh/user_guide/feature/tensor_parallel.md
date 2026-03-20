# Tensor Parallel

TP（Tensor Parallel，张量并行）是一种模型并行的策略，它通过将张量（如权重矩阵、激活值等）在多个设备（如NPU）之间进行切分 ，从而实现模型的分布式推理。

## 限制与约束<a name="section94261452203419"></a>

-   Atlas 800I A2 推理服务器和Atlas 800I A3 超节点服务器支持此特性。
-   DeepSeek-V3和DeepSeek-R1模型支持“Lmhead矩阵local tp切分”、“O project矩阵local tp切分”、“tp大于1”。
-   PD分离且D节点是分布式的场景，支持Lmhead矩阵local tp切分和O project矩阵local tp切分，减少矩阵计算时间，降低推理时延。
-   PD分离且D节点是分布式低时延场景，当tp大于1时支持MLA的tp切分，小batch低时延场景能减少decode推理时延。
-   “tp”大于1时，不支持和O project矩阵local tp切分同时开启，也不建议和Lmhead矩阵local tp同时开启。

## 参数说明

开启“Lmhead矩阵local tp切分”，需要配置的参数如[表1](#table1)所示。

**表 1**  Lmhead矩阵local tp切分补充参数：**ModelConfig中的models参数** <a id="table1"></a>

|配置项|取值类型|取值范围|配置说明|
|--|--|--|--|
|deepseekv2 | 
|parallel_options|
|lm_head_local_tp|int|[1，worldSize / 节点数]|表示LmHead张量并行切分数。<br><ul><li>仅DeepSeek-R1、DeepSeek-V3和DeepSeek-V3.1模型支持此特性。</li><li>默认值：-1。表示不开启切分</li></ul>|


开启“O project矩阵local tp切分”，需要配置的参数如[表2](#table2)所示。

**表 2**  O project矩阵local tp切分补充参数：**ModelConfig中的models参数**
<a id="table2"></a>
|配置项|取值类型|取值范围|配置说明|
|--|--|--|--|
|deepseekv2|
|parallel_options|
|o_proj_local_tp|int|[1，worldSize / 节点数]|表示Attention O矩阵切分数。<br><ul><li>仅DeepSeek-R1、DeepSeek-V3和DeepSeek-V3.1模型支持此特性。</li><li>默认值：-1，表示不开启切分</li></ul>|


## 执行推理

1.  打开Server的config.json文件。

    ```bash
    cd {MindIE安装目录}/latest/mindie-service/
    vi conf/config.json
    ```

2.  配置服务化参数。在Server的config.json文件按照[表1](#table1)和[表2](#table2)添加相应参数， 服务化参数说明请参见[配置参数说明（服务化）](../user_manual/service_parameter_configuration.md)章节，参数配置示例如下。

    下面以DeepSeek-R1模型为例。下方以开启tp切分，关闭Lmhead矩阵local tp切分和O project矩阵local tp切分为例示意。

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
             "tp": 2
             "models": {
                "deepseekv2": {
                    "parallel_options": {
                        "lm_head_local_tp": -1,
                        "o_proj_local_tp": -1,
                    }
                }
             }
          }
       ]
    },
    ```

3.  启动服务。

    ```bash
    ./bin/mindieservice_daemon
    ```

