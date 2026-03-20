# Context Parallel

Context Parallel（CP，上下文并行）主要针对Self-attention模块在sequence维度进行并行计算。CP通过将长序列在上下文维度进行切分，分配到不同设备并行处理，减少首token响应时间。CP实现主要包括：

1. 各个设备计算各自的attention，设备之间用ring的方式传递KV值来获得分块运算的结果，整体原理类似ring-attention。
2. 用Flash-attention 2算法进行分块运算， 最后对分块结果进行修正。

## 限制与约束

-  Atlas 800I A2 推理服务器和Atlas 800I A3 超节点服务器支持此特性。
-  当前仅DeepSeek-R1的W8A8量化模型、DeepSeek-R1的W4A8量化模型、 DeepSeek-V3的W4A8量化模型和DeepSeek-V3.1的W4A8量化模型支持此特性。
-  当前不支持CP单独开启，开启CP需要同时开始SP。
-  支持PD分离场景和PD混部场景。
-  PD混部场景时：
    -  该特性可以和SP(sequence parallel)、TP(tensor parallel)同时使用。开启CP特性时，DP(data parallel)必须等于1，SP必须等于TP，且CP、DP和TP的乘积等于Worldsize。
    -  该特性支持与MTP=1、异步调度、prefixcache特性叠加使用。

-  PD分离场景时：
    -  仅支持在P节点开启CP特性，该特性可以和SP、TP、MTP同时使用。开启CP特性时，DP必须等于1，SP必须等于TP，且CP、DP和TP的乘积等于Worldsize。
    -  该特性支持与MTP、异步调度、prefixcache特性叠加使用。

-  该特性不支持BF16。

## 参数说明

开启CP特性，需要配置的服务化参数如[表1](#ModelConfig参数)。

**表 1**  补充参数：**ModelDeployConfig中的ModelConfig参数** <a id="ModelConfig参数"></a>

|配置项|取值类型|取值范围|配置说明|
|--|--|--|--|
|cp|int|[1，2]|将一个输入序列切分后得到的份数。<br>1：不开启CP特性。<br>2：输入序列切分成2份。<br>目前开启CP特性，切分的份数仅支持“2”。|


## 执行推理

1. 打开Server的config.json文件。

    ```bash
    cd {MindIE安装目录}/latest/mindie-service/
    vi conf/config.json
    ```

2. 配置服务化参数。在Server的config.json文件添加“cp“字段（以下加粗部分），参数字段解释请参见[参数说明](#参数说明)。服务化参数说明请参见[配置参数说明（服务化）](../user_manual/service_parameter_configuration.md)章节，参数配置示例如下。

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
                "worldSize" : 16,
                "cpuMemSize" : 5,
                "npuMemSize" : -1,
                "backendType" : "atb",
                "trustRemoteCode" : false,
                "dp": 1,
                "cp": 2,
                "sp": 8,
                "tp": 8,
                "moe_ep": 16,
                "moe_tp": 1
            }
        ]
    }
    ```

3. 启动服务。

    ```
    ./bin/mindieservice_daemon
    ```

