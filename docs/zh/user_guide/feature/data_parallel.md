# Data Parallel

Data Parallel（DP，数据并行）将推理请求划分为多个批次，并将每个批次分配给不同的设备进行并行处理，每部分设备都并行处理不同批次的数据，然后将结果合并。

## 使用场景
在显存足够时，均可开启数据并行特性，以提高吞吐。

## 限制与约束

-  Atlas 800I A2 推理服务器和Atlas 800I A3 超节点服务器支持此特性。
-  所有模型的Attention模块、MLP模块。
-  数据并行支持同张量并行在同一模块上叠加使用。

## 参数说明

开启数据并行特性，需要配置的补充参数如[表1](#table1)所示。

**表 1**  数据并行特性补充参数：**ModelDeployConfig中的ModelConfig参数** <a id="table1"></a>

|配置项|取值类型|取值范围|配置说明|
|--|--|--|--|
|tp|int32_t|<ul><li>未配置dp或dp值为-1时：取值为worldSize参数值。</li><li>与dp配合使用时：tp*dp的值必须等于worldSize参数值。</li></ul><br>例：若worldSize为8，dp配置为2，则tp的值只能配置为4。|整网张量并行数。<br>选填，默认值为设置的worldSize参数值。|
|dp|int32_t|<ul><li>不执行该并行方式时：-1</li><li>与tp配合使用时：dp*tp的值必须等于worldSize参数值。</li></ul><br>例：若worldSize为8，tp配置为4，则dp的值只能配置为2。|Attention模块中的数据并行数。<br>选填，默认值：-1，表示不执行数据并行。|
|cp|int32_t|<ul><li>不执行该并行方式时：1</li><li>与sp配合使用时：dp\*tp\*cp的值必须等于worldSize参数值，且dp必须为1。</li></ul><br>例：若worldSize为16，tp配置为8，sp配置为8，dp的值只能配置为1，cp的值只能配置为2。|选填，默认值：1，表示不执行上下文并行。<br>Attention模块中的上下文并行数。|
|sp|int32_t|<ul><li>不执行该并行方式时：1</li><li>与tp配合使用时：sp的值必须等于tp的参数值。</li></ul><br>例：若worldSize为16，tp配置为8，dp配置为2，sp的值只能配置为8。|选填，默认值：1，表示不执行序列并行。<br>Attention模块中的序列并行数。|


> [!NOTE]说明 
> 不配置以上补充参数时，推理过程中默认使用tp和moe\_tp并行方式。

## 执行推理

已在环境上安装CANN和MindIE详情请参见《MindIE安装指南》。

1. 设置优化显存分配的环境变量

    ```bash
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
    ```

2. 打开Server的config.json文件。

    ```bash
    cd {MindIE安装目录}/latest/mindie-service/
    vi conf/config.json
    ```

3. 配置服务化参数。在Server的config.json文件按照[表1](#table1)添加相应参数，服务化参数说明请参见[配置参数说明（服务化）](../user_manual/service_parameter_configuration.md)章节，参数配置示例如下。

    ```json
    "ModelConfig" : [
        {
            "modelInstanceType" : "Standard",
            "modelName" : "deepseekv2",
            "modelWeightPath" : "/home/data/DeepSeek-V2-Chat-W8A8-BF16/",
            "worldSize" : 8,
            "cpuMemSize" : 5,
            "npuMemSize" : 1,
            "backendType" : "atb",
            "trustRemoteCode" : false,
            "tp": 1,
            "dp": 8,
            "cp": 1,
            "sp": 1
        }
    ]
    ```

    以上参数设置表明使用8卡进行推理，Attention模块使用数据并行，MoE模型使用张量并行。

4. 启动服务。

    ```
    ./bin/mindieservice_daemon
    ```

5. 发送推理请求。具体请参考《MindIE Motor开发指南》中的“集群管理组件 \> 调度器（Coordinator） \> RESTful接口API \> 用户侧接口 \> OpenAI推理接口”章节。

