# 思考预算

部分大模型输出结果时会附带其思考过程，本特性旨在控制模型思考深度，当思考内容超过设定的thinking\_budget时，系统会使用提示词对思考过程进行截断，促使模型提前结束思考。该特性适用于在响应速度与答案质量之间灵活权衡的场景。

## 限制与约束

- Atlas 800I A2 推理服务器、Atlas 800I A3 超节点服务器和Atlas 300I Duo 推理卡支持此特性。
- 当前仅Qwen3-32B、Qwen3-235B-A22B和Qwen3-30B-A3B模型支持此特性。
- 开启thinking_budget需在请求中传入如下字段："chat\_template\_kwargs": \{"thinking\_budget":  _<uint32_t\>_\}, 取值范围为[1, MAX_UINT32_T]。
- 当前仅支持OpenAI推理接口。
- 该特性暂不支持与use_beam_search等多序列推理相关的后处理参数同时开启。

## 参数说明

开启思考预算特性，需要配置的参数如[表1](#table1)所示。

**表 1**  思考预算特性补充参数：**ModelConfig中的models参数** <a id="table1"></a>

|配置项|取值类型|长度范围|配置说明|
|--|--|--|--|
|early_stopping_text|string|[1,1024]|结束思考提示词。<br>开启thinking_budget后，当模型思考输出超过预算后使用该提示词进行截断。<br>不同模型提示词不同，qwen系列模型提示词参考下面的参数配置示例。|

## 执行推理

1. 打开Server的config.json文件。

    ```bash
    cd /usr/local/lib/python3.11/site-packages/mindie_llm
    vi conf/config.json
    ```

2. 配置服务化参数。按照[表1](#table1)在Server的config.json文件中添加“early\_stopping\_text”字段，服务化参数说明请参见[配置参数说明（服务化）](../user_manual/service_parameter_configuration.md)章节，参数配置示例如下。

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
                                "qwen3": {"early_stopping_text": "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"}
                        }
                    }
                ]
            },
    ```

    > [!NOTE]说明
    >- Qwen3-30B-A3B模型："qwen3"字段应修改为"qwen3\_moe"。
    >- 在thinking_budget设置过低的情况下，推理结果有概率切换到与提示词相同的语言。

3. 启动服务。PD混部场景可请参考《MindIE Motor开发指南》中的“快速入门 \> [启动服务](https://gitcode.com/Ascend/MindIE-Motor/blob/dev/docs/zh/user_guide/quick_start.md)”章节，
PD分离场景可参考《MindIE Motor开发指南》中的“集群服务部署 \> [PD分离服务部署](https://gitcode.com/Ascend/MindIE-Motor/blob/master/docs/zh/user_guide/service_deployment/pd_separation_service_deployment.md)”章节。
4. 发送请求。参数说明见《MindIE LLM开发指南》中的“[服务化接口使用指导](https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/user_manual/service_APIs_usage_guidance.md)”章节。
