# 思考解析

部分大模型输出结果包含思考过程，本特性旨在对大模型的输出内容进行结构化解析，将模型在推理过程中产生的“思考过程（think）”与最终的“输出结果（content）”进行分离，并分别存储于"reasoning\_content"和"content"两个字段中。

- reasoning\_content：用于存储模型在生成回答前的推理、分析、逻辑判断等内部思维过程。
- content：用于存储模型最终对外输出的回答或决策结果。

## 限制与约束

- Atlas 800I A2 推理服务器、Atlas 800I A3 超节点服务器和Atlas 300I Duo 推理卡支持此特性。
- 当前仅Qwen3-32B、Qwen3-235B-A22B、Qwen3-30B-A3B、DeepSeek-R1和DeepSeek-V3.1模型支持此特性。
- DeepSeek-V3.1模型开启思考解析时，需在请求中传入如下字段："chat\_template\_kwargs": \{"enable\_thinking":  _<bool\>_\}，或者在tokenizer\_config.json中添加"enable\_thinking": <bool\>
- 当前仅支持OpenAI推理接口。

## 参数说明

开启思考解析特性，需要配置的参数如[表1](#table1)所示。

**表 1**  思考解析特性补充参数：**ModelConfig中的models参数** <a id="table1"></a>

|配置项|取值类型|取值范围|配置说明|
|--|--|--|--|
|enable_reasoning|bool|true<br>false|是否开启模型思考解析，将输出分别解析为“reasoning_content”和“content”两个字段。false：关闭<br>true：开启<br>必填，默认值：false。|

## 执行推理

1. 打开Server的config.json文件。

    ```bash
    cd {MindIE安装目录}/latest/mindie-service/
    vi conf/config.json
    ```

2. 配置服务化参数。按照[表1](#table1)在Server的config.json文件中添加“enable\_reasoning”字段，服务化参数说明请参见[配置参数说明（服务化）](../user_manual/service_parameter_configuration.md)章节，参数配置示例如下。

    以Qwen3-32B为例：

    ```json
     "ModelDeployConfig" :
            {
                "maxSeqLen" : 2560,
                "maxInputTokenLen" : 2048,
                "truncation" : 0,
                "ModelConfig" : [
                    {
                        "modelInstanceType" : "Standard",
                        "modelName" : "Qwen3-32B",
                        "modelWeightPath" : "/data/weight/Qwen3-32B",
                        "worldSize" : 1,
                        "cpuMemSize" : 0,
                        "npuMemSize" : -1,
                        "backendType" : "atb",
                        "trustRemoteCode" : false,
                        "async_scheduler_wait_time": 120,
                        "kv_trans_timeout": 10,
                        "kv_link_timeout": 1080,
                        "models": {
                                "qwen3": {"enable_reasoning": true}
                        }
                    }
                ]
            },
    ```

    > [!NOTE]说明
    >- Qwen3-30B-A3B模型："qwen3"字段应修改为"qwen3\_moe"。
    >- DeepSeek-R1模型："qwen3"字段应修改为"deepseekv2"，并将DeepSeek-R1权重文件中的"model\_type"字段修改为"deepseek\_v3"
    >- DeepSeek-V3.2 模型：`"qwen3"` 字段应修改为 `"deepseek_v32"`

3. 启动服务。具体请参考《MindIE Motor开发指南》中的“快速入门 \> [启动服务](https://gitcode.com/Ascend/MindIE-Motor/blob/dev/docs/zh/user_guide/quick_start.md)”章节。
4. 发送请求。参数说明见《MindIE Motor开发指南》中的“集群管理组件 \> 调度器（Coordinator） \> RESTful接口API \> 用户侧接口 \> OpenAI推理接口”章节。
