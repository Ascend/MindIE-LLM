# Multi-LoRA

LoRA（Low-Rank Adaptation）是一种高效的参数微调方法。将大模型的权重矩阵分解为原始权重矩阵和两个低秩矩阵的乘积，即W' = W + BA。由于B和A矩阵参与训练的参数量远小于原始权重，其乘积结果又能合入线性层并向下传递，从而达到大模型轻量级微调的目的。

Multi-LoRA指基于一个基础模型，使用多个不同的LoRA权重进行推理。每个请求带有指定的LoRA ID，推理时动态匹配对应的LoRA权重。部署服务时，LoRA权重和基础模型权重预先加载至显存中。一个推理请求至多使用一个LoRA权重，兼容推理请求不使用LoRA权重的情况。对于大参数量的模型，若模型参数量过大，无法单卡加载时，可进行Tensor Parallel并行。

LoRA权重中需包含"adapter\_config.json"和"adapter\_model.safetensors"文件，文件描述如[表1](#table1)所示。

**表 1**  LoRA权重的文件说明  <a id="table1"></a>

|文件名称|文件描述|示例|
|--|--|--|
|adapter_config.json|包含LoRA权重的超参。|r（LoRA微调中的秩大小），rank_pattern，lora_alpha（LoRA低秩矩阵的缩放系数）和alpha_pattern。|
|adapter_model.safetensors|包含权重，权重以键值对的形式保存，其中LoRA权重的键名在基础模型的键名前后增加"base_model.model"前缀和"lora_A.weight"、"lora_B.weight"后缀。|基础模型中的键名为"model.layers.9.self_attn.v_proj.weight"时，则LoRA权重中对应的键名为："base_model.model.model.layers.9.self_attn.v_proj.lora_A.weight"和"base_model.model.model.layers.9.self_attn.v_proj.lora_B.weight"。|


## 限制与约束

-  Atlas 800I A2 推理服务器、Atlas 800I A3 超节点服务器和Atlas 300I Duo 推理卡支持此特性。
-  LoRA权重个数上限受硬件显存限制，建议数量为小于等于10个。
-  仅在ATB Models使用Python组图时支持LoRA权重动态加载和卸载。
-  支持线性层携带LoRA权重。
-  不支持和量化、PD分离、并行解码、SplitFuse、MTP、异步调度、Micro Batch以及Prefix Cache特性同时开启。
-  仅Qwen2.5-7B、Qwen2.5-14B、Qwen2.5-32B、Qwen2.5-72B、Qwen3-32B、LLaMA3.1-8B、LLaMA3.1-70B和Qwen2-72B支持该特性。
-  LoRA权重名称长度不能超过256个字符。
-  仅支持vLLM、TGI和vLLM兼容OpenAI接口。

## 参数说明

开启Multi-LoRA特性，需要配置的服务化参数如[表2](#table2)所示。

**表 2**  Multi-LoRA特性补充参数：**ModelDeployConfig中的参数**  <a id="table2"></a>

|配置项|取值类型|取值范围|配置说明|
|--|--|--|--|
|maxLoras|uint32_t|上限根据显存和用户需求来决定，最小值需大于0。|最大可加载的LoRA数量。选填，开启LoRA权重动态加载和卸载时需配置。默认值为0。|
|maxLoraRank|uint32_t|上限根据显存和用户需求来决定，最小值需大于0。|可加载LoRA权重最大的秩。选填，开启LoRA权重动态加载和卸载时需配置。默认值为0。|
|LoraModules|
|name|string|由大写字母、小写字母、数字、中划线和下划线组成，且不以中划线和下划线作为开头和结尾，字符串长度小于或等于256。|必填，LoRA ID。|
|path|string|文件绝对路径长度的上限与操作系统的设置（Linux为PATH_MAX）有关，最小值为1。|必填，LoRA权重路径。该路径会进行安全校验，需要和执行用户的属组和权限保持一致。|
|baseModelName|string|由大写字母、小写字母、数字、中划线、点和下划线组成，且不以中划线、点和下划线作为开头和结尾，字符串长度小于或等于256。|必填，基础模型名称。与ModelConfig参数说明中的modelName参数保持一致。|


## 执行推理

-   纯模型使用：

    已在环境上安装CANN和ATB Models，详情请参见《MindIE安装指南》。

    > [!NOTE]说明 
    > 本次样例参考以下安装路径进行：
    > 安装ATB Models并初始化ATB Models环境变量。模型仓set\_env.sh脚本中有初始化“\$\{ATB\_SPEED\_HOME\_PATH\}”环境变量的操作，所以source模型仓中set\_env.sh脚本时会同时初始化“\$\{ATB\_SPEED\_HOME\_PATH\}”环境变量。

    以LLaMA3.1-70B为例，下载基础模型和LoRA权重后，您可以使用以下指令执行对话测试，共3个请求组成Batch进行推理，每个推理请求中LoRA权重不同。run\_pa脚本参数参考[表2](../user_manual/offline_inference.md#table2)章节。

    通过“lora\_modules”指定基础模型和LoRA权重的绑定关系：

    -  权重名称为权重的别名，长度不能超过256个字符，在后续请求中用于指定Lora权重进行推理。
    -  支持配置多个LoRA权重。

    ```bash
    cd ${ATB_SPEED_HOME_PATH}
    torchrun --nproc_per_node 8 --master_port 20030 -m examples.run_pa --model_path {基础模型权重} --max_output_length 20 --max_batch_size 3 --input_dict '[{"prompt": "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?", "adapter": "{Lora权重1的名称}"}, {"prompt": "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?", "adapter": "{Lora权重2的名称}"}, {"prompt": "What is deep learning?", "adapter": "base"}]' --lora_modules '{"{Lora权重1的名称}": "{Lora权重1的路径}", "{Lora权重2的名称}": "{Lora权重2路径}"}'
    ```

-  服务化使用：

    > [!NOTE]说明 
    > lora\_adapter.json文件配置方式已日落，新的配置方式是在MindIE Motor的config.json文件中添加LoraModules字段开启Multi LoRA特性，详细操作步骤如下所示。

    以LLaMA3.1 70B模型为例，简单介绍Multi LoRA如何使用。

    1. 打开Server的config.json文件。

        ```bash
        cd {MindIE安装目录}/latest/mindie-service/
        vi conf/config.json
        ```

    2. 配置服务化参数。在Server的config.json文件添加maxLoras、maxLoraRank以及LoraModules字段（以下加粗部分），参数字段说明请参见[表2](#table2)，服务化参数说明请参见[配置参数说明（服务化）](../user_manual/service_parameter_configuration.md)章节，参数配置示例如下。

        ```json
        {    
            "ServerConfig" :
            {
                "ipAddress" : "127.0.0.1",
                "managementIpAddress" : "127.0.0.2",
                "port" : 1025,
                "managementPort" : 1026,
            },
            "BackendConfig": {
                "backendName" : "mindieservice_llm_engine",
                "modelInstanceNumber" : 1,
                "npuDeviceIds" : [[0,1,2,3,4,5,6,7]],
                "tokenizerProcessNumber" : 8,
                "multiNodesInferEnabled": false,
                "multiNodesInferPort": 1120,
                "interNodeTLSEnabled": true,
                "interNodeTlsCaPath": "security/grpc/ca/",
                "interNodeTlsCaFiles": ["ca.pem"],
                "interNodeTlsCert": "security/grpc/certs/server.pem",
                "interNodeTlsPk": "security/grpc/keys/server.key.pem",
                "interNodeTlsCrlPath" : "security/grpc/certs/",
                "interNodeTlsCrlfiles" : ["server_crl.pem"],
                "ModelDeployConfig":
                {
                    "maxSeqLen" : 2560,
                    "maxInputTokenLen" : 2048,
                    "truncation" : 0,
                    "ModelConfig" : [
                        {
                            "modelInstanceType": "Standard",
                            "modelName" : "llama3.1-70b",
                            "modelWeightPath" : "/data/weights/llama3.1-70b-safetensors",
                            "worldSize" : 8,
                            "cpuMemSize" : 5,
                            "npuMemSize" : -1,
                            "backendType": "atb",
                            "trustRemoteCode": false
                        }
                    ],
                    "maxLoras": 4,
                    "maxLoraRank": 296,
                    "LoraModules" :[{
                    "name" : "adapter1",
                    "path" : "/data/lora_model_weights/llama3.1-70b-lora",
                    "baseModelName" : "llama3.1-70b"
                    }]
                }, 
            }
        }
        ```

    3. 启动服务。

        ```bash
        ./bin/mindieservice_daemon
        ```

    4. 动态加载、卸载或查询LoRA。

        加载请求：

        ```bash
        curl -X POST http://127.0.0.2:1026/v1/load_lora_adapter \
        -H "Content-Type: application/json" \
        -d '{
            "lora_name": "adapter2",
            "lora_path": "/data/lora_model_weights/llama3.1-70b-lora"
        }'
        ```

        卸载请求：

        ```bash
        curl -X POST 127.0.0.2:1026/v1/unload_lora_adapter -d '{
            "lora_name": "adapter2"
        }'
        ```

        查询请求：

        ```bash
        curl http://127.0.0.1:1025/v1/models
        ```

    5. 使用以下指令发送请求。

        其中"model"参数可以设置为基础模型名称（config.json配置文件中"ModelConfig"字段下的"modelName"参数的值）或LoRA ID（config.json配置文件中"LoraModules"字段下"name"参数的值）。当"model"参数为基础模型名称时，不使用Lora权重进行推理。当"model"参数为LoRA ID时，启用基础模型权重和指定的LoRA权重进行推理。

        ```bash
        curl https://127.0.0.1:1025/generate \
        -H "Content-Type: application/json" \
        --cacert ca.pem --cert client.pem  --key client.key.pem \
        -X POST \
        -d '{
        "model": "${基础模型名称}",
        "prompt": "Taxation in Puerto Rico -- The Commonwealth government has its own tax laws and Puerto Ricans are also required to pay some US federal taxes, although most residents do not have to pay the federal personal income tax. In 2009, Puerto Rico paid $3.742 billion into the US Treasury. Residents of Puerto Rico pay into Social Security, and are thus eligible for Social Security benefits upon retirement. However, they are excluded from the Supplemental Security Income.\nQuestion: is federal income tax the same as social security?\nAnswer:",
        "max_tokens": 20,
        "temperature": 0
        }'
        
        curl https://127.0.0.1:1025/generate \
        -H "Content-Type: application/json" \
        --cacert ca.pem --cert client.pem  --key client.key.pem \
        -X POST \
        -d '{
        "model": "adapter1",
        "prompt": "Taxation in Puerto Rico -- The Commonwealth government has its own tax laws and Puerto Ricans are also required to pay some US federal taxes, although most residents do not have to pay the federal personal income tax. In 2009, Puerto Rico paid $3.742 billion into the US Treasury. Residents of Puerto Rico pay into Social Security, and are thus eligible for Social Security benefits upon retirement. However, they are excluded from the Supplemental Security Income.\nQuestion: is federal income tax the same as social security?\nAnswer:",
        "max_tokens": 20,
        "temperature": 0
        }'
        ```

