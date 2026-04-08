# Sequence Parallel

Sequence Parallel（SP，序列并行）通过对KV Cache进行切分，使得每个sprank保存的KV Cache各不相同，达到节省显存，支持长序列的功能。

## 限制与约束

-  Atlas 800I A2 推理服务器和Atlas 800I A3 超节点服务器支持此特性。
-  当前仅DeepSeek-R1的W8A8量化模型、DeepSeek-R1的W4A8量化模型、 DeepSeek-V3的W4A8量化模型和DeepSeek-V3.1的W4A8量化模型支持此特性。
-  支持PD分离场景和PD混部场景。
-  SP必须等于TP。
-  PD混部场景时：
    -  该特性可以和DP(data parallel)、TP(tensor parallel)同时使用，DP和TP的乘积等于Worldsize。
    -  该特性可以和CP(context parallel)、TP、MTP同时使用，CP和TP的乘积等于Worldsize。
    -  该特性支持与MTP=1、异步调度、Prefix Cache特性叠加使用。

-  PD分离场景时：
    -  仅支持在P节点开启SP特性，该特性可以和DP、TP、MTP同时使用 ，DP和TP的乘积等于Worldsize。
    -  仅支持在P节点开启SP特性，该特性可以和CP、TP、MTP同时使用，CP和TP的乘积等于Worldsize。
    -  该特性支持与MTP、异步调度、Prefix Cache特性叠加使用。

-  该特性不支持BF16。

## 参数说明

开启SP特性，需要配置的服务化参数如[表1](#table1)所示。

**表 1**  SP特性补充参数：**ModelDeployConfig中的ModelConfig参数**  <a id="table1"></a>

|配置项|取值类型|取值范围|配置说明|
|--|--|--|--|
|sp|int|sp=tp|KV Cache切分得到的份数。|


## 执行推理

1. 打开Server的config.json文件。

    ```bash
    cd {MindIE安装目录}/latest/mindie-service/
    vi conf/config.json
    ```

2. 配置服务化参数。在Server的config.json文件添加“sp“字段，参数字段解释请参见[表1](#table1)。config.json文件的详细配置说明，PD分离场景请参考《MindIE Motor开发指南》中的“集群服务部署 \> PD分离服务部署”章节；PD混部场景请参考_《MindIE安装指南》中的“配置MindIE \> 配置Server \> 多机推理”章节_。

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
                "dp": 2,
                "sp": 8,
                "tp": 8,
                "moe_ep": 16,
                "moe_tp": 1
            }
        ]
    }
    ```

3.  启动服务。

    ```bash
    ./bin/mindieservice_daemon
    ```

