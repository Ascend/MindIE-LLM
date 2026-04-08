# A3单机PD混合部署

# 一、服务化部署流程

1. 安装最新MindIE转测镜像：加载镜像创建对应的容器，多机每台机器都需要安装
2. 修改服务化配置文件：`vim /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json`，多机每台机器容器对应的该文件都需要修改
3. 拉起服务：脚本中包含所有环境变量，多机每个机器都要执行对应的脚本
4. 发送aisbench命令：修改aisbench的精度数据集、模型对应python脚本，发送aisbench精度或性能测试指令
5. 对比精度性能基线

# 二、服务化配置文件

需要更改（新增）的参数如下：

| 参数名                 | 原始值            | 应修改值                     | 备注                |
| ---------------------- | ---------------- | --------------------------- | ------------------- |
| `httpsEnabled`           | `true`                                 | `false`                                                   |                                          |
| `tokenTimeout`           | `600`                                  | `3600`                                                    |                                          |
| `e2eTimeout`             | `600`                                  | `3600`                                                    |                                          |
| `npuDeviceIds`           | `[[0,1,2,3]]`                          | `[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]`|                                          |
| `multiNodesInferEnabled` | `false`                                | `多机为true，单机为默认值`                                | 多机没有改会识别为worldsize只有16        |
| `interNodeTLSEnabled`    | `true`                                 | `多机为false，单机为默认值`                               |                                          |
| `modelName`              | `llama_65b`                            | `deepseek`                                                | 自定义字段，aisbench等请求的模型名需一致 |
| `modelWeightPath`        | `/data/weights/llama1-65b-safetensors` | `/path/to/file`                                           | 权重绝对路径                             |
| `worldSize`              | `4`                                    | `16`                                                      |                                          |
| `dp`                     | `新增`                                 | `2`                                                       |                                          |
| `cp`                     | `新增`                                 | `1`                                                       |                                          |
| `tp`                     | `新增`                                 | `8`                                                       |                                          |
| `sp`                     | `新增`                                 | `1`                                                       |                                          |
| `moe_tp`                 | `新增`                                 | `4`                                                       |                                          |
| `moe_ep`                 | `新增`                                 | `4`                                                       |                                          |
| `plugin_params`          | `新增`                                 | "{\"plugin_type\":\"mtp\",\"num_speculative_tokens\": 1}" |                                          |
| `maxPrefillBatchSize`    | `50`                                   | `2`                                                       |                                          |
| `maxPrefillTokens`       | `8192`                                 | `16384`                                                   |                                          |
| `maxIterTimes`           | `512`                                  | `16384`                                                   |                                          |

多机的config配置文件应该保证每台机器完全一样，此外还有一个`models`参数的增加，请在下面完整配置文件中查看：

```json
{
    "Version" : "1.0.0",

    "ServerConfig" :
    {
        "ipAddress" : "127.0.0.2",
        "managementIpAddress" : "127.0.0.2",
        "port" : 1025,
        "managementPort" : 1026,
        "metricsPort" : 1027,
        "allowAllZeroIpListening" : false,
        "maxLinkNum" : 1000,
        "httpsEnabled" : false,
        "fullTextEnabled" : false,
        "tlsCaPath" : "security/ca/",
        "tlsCaFile" : ["ca.pem"],
        "tlsCert" : "security/certs/server.pem",
        "tlsPk" : "security/keys/server.key.pem",
        "tlsCrlPath" : "security/certs/",
        "tlsCrlFiles" : ["server_crl.pem"],
        "managementTlsCaFile" : ["management_ca.pem"],
        "managementTlsCert" : "security/certs/management/server.pem",
        "managementTlsPk" : "security/keys/management/server.key.pem",
        "managementTlsCrlPath" : "security/management/certs/",
        "managementTlsCrlFiles" : ["server_crl.pem"],
        "metricsTlsCaFile" : ["metrics_ca.pem"],
        "metricsTlsCert" : "security/certs/metrics/server.pem",
        "metricsTlsPk" : "security/keys/metrics/server.key.pem",
        "metricsTlsCrlPath" : "security/metrics/certs/",
        "metricsTlsCrlFiles" : ["server_crl.pem"],
        "inferMode" : "standard",
        "interCommTLSEnabled" : true,
        "interCommPort" : 1121,
        "interCommTlsCaPath" : "security/grpc/ca/",
        "interCommTlsCaFiles" : ["ca.pem"],
        "interCommTlsCert" : "security/grpc/certs/server.pem",
        "interCommPk" : "security/grpc/keys/server.key.pem",
        "interCommTlsCrlPath" : "security/grpc/certs/",
        "interCommTlsCrlFiles" : ["server_crl.pem"],
        "openAiSupport" : "vllm",
        "tokenTimeout" : 3600,
        "e2eTimeout" : 3600,
        "distDPServerEnabled":false
    },

    "BackendConfig" : {
        "backendName" : "mindieservice_llm_engine",
        "modelInstanceNumber" : 1,
        "npuDeviceIds" : [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]],
        "tokenizerProcessNumber" : 8,
        "multiNodesInferEnabled" : false,
        "multiNodesInferPort" : 1120,
        "interNodeTLSEnabled" : true,
        "interNodeTlsCaPath" : "security/grpc/ca/",
        "interNodeTlsCaFiles" : ["ca.pem"],
        "interNodeTlsCert" : "security/grpc/certs/server.pem",
        "interNodeTlsPk" : "security/grpc/keys/server.key.pem",
        "interNodeTlsCrlPath" : "security/grpc/certs/",
        "interNodeTlsCrlFiles" : ["server_crl.pem"],
        "kvPoolConfig" : {"backend":"", "configPath":""},
        "ModelDeployConfig" :
        {
            "maxSeqLen" : 24576,
            "maxInputTokenLen" : 4096,
            "truncation" : false,
            "ModelConfig" : [
                {
                    "modelInstanceType" : "Standard",
                    "modelName" : "deepseek",
                    "modelWeightPath" : "/path/to/file",
                    "worldSize" : 16,
                    "cpuMemSize" : 5,
                    "npuMemSize" : -1,
                    "backendType" : "atb",
                    "trustRemoteCode" : false,
		            "dp": 2,
                    "cp": 1,
                    "tp": 8,
                    "sp": 1,
                    "moe_tp": 4,
                    "moe_ep": 4,
                    "plugin_params": "{\"plugin_type\":\"mtp\",\"num_speculative_tokens\": 1}",
                    "models": {
                        "deepseekv2": {
                            "ep_level":1,
                            "kv_cache_options": {"enable_nz": true},
                            "enable_mlapo_prefetch": true
			    }
		    }
                }
            ]
        },

        "ScheduleConfig" :
        {
            "templateType" : "Standard",
            "templateName" : "Standard_LLM",
            "cacheBlockSize" : 128,

            "maxPrefillBatchSize" : 2,
            "maxPrefillTokens" : 16384,
            "prefillTimeMsPerReq" : 150,
            "prefillPolicyType" : 0,

            "decodeTimeMsPerReq" : 50,
            "decodePolicyType" : 0,

            "maxBatchSize" : 200,
            "maxIterTimes" : 16384,
            "maxPreemptCount" : 0,
            "supportSelectBatch" : false,
            "maxQueueDelayMicroseconds" : 5000,
            "maxFirstTokenWaitTime": 2500
        }
    }
}
```
该配置文件只需要改权重路径，建议新建文件直接复制粘贴修改后再覆盖容器内部原文件



# 三、拉起脚本与环境变量

单机拉起服务化脚本：在容器内任意处`bash A3_single.sh`执行即可拉起服务，A3_single.sh脚本在MindIE-Motor/mindie_service/server/scripts/A3_single.sh查看。

多机拉起服务化脚本：多机需要同时执行脚本，且多机的每台机器执行的脚本`MIES_CONTAINER_IP`参数不同。双机对比单机脚本多出ranktable路径、主节点ip、当前节点ip几个环境变量，且这几个环境变量需要自行指定。


**环境变量说明**
- NPU_MEMORY_FRACTION：参数为显存比例因子，默认参数为0.92，高并发场景下建议 <=0.92。
    - 建议配置方案：建议将该值设置为可拉起服务的最小值。具体方法是，按照默认配置启动服务，若无法拉起服务，则上调参数至可拉起为止；若拉起服务成功，则下调该参数至刚好拉起服务为止。总之，在服务能正常拉起的前提下，更低的值可以保障更高的服务系统稳定性。

# 四、aisbench测精度与性能

<span style="color: red; font-weight: bold; font-size: 23px;">最新MindIE镜像，aisbench工具无需安装直接使用，且全面覆盖benchmark</span>

随意在同一个网段的一个机器中起一个最新MindIE版本的容器，aisbench工具已经装好在`/opt/package/benchmark/ais_bench/`路径中，请勿自己安装，需要改几个python配置文件，然后发送aisbench命令即可，需要将数据集上传到`/opt/package/benchmark/ais_bench/datasets`路径中。

### 4.1 修改vllm_api_general_chat.py

```shell
cd /opt/package/benchmark/ais_bench/benchmark/configs/models/vllm_api/
vim vllm_api_general_chat.py
```

```
from ais_bench.benchmark.models import VLLMCustomAPIChat
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="/path/to/file",
        model="deepseek",
        max_seq_len=24576,
        request_rate = 0,
        retry = 2,
        host_ip = "",
        host_port = 1025,
        max_out_len = 20480,
        batch_size=30,
        generation_kwargs = dict(
            temperature = 0.6,
            top_k = 10,
            top_p = 0.95,
            seed = None,
            repetition_penalty = 1.03,
        )
    )
]
```

<span style="color: red; font-weight: bold; font-size: 18px;">**跑不同并发数和请求频率的性能，在此处修改batch_size和request_rate**
</span>

### 4.2 修改vllm_api_stream_chat.py

```
cd /opt/package/benchmark/ais_bench/benchmark/configs/models/vllm_api/
vim vllm_api_stream_chat.py
```

```
from ais_bench.benchmark.models import VLLMCustomAPIChatStream
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="/path/to/file",
        model="deepseek",
        request_rate = 0.96,
        retry = 2,
        host_ip = "141.61.105.123",
        host_port = 1025,
        max_out_len = 512,
        batch_size=400,
        generation_kwargs = dict(
            temperature = 0.5,
            top_k = 10,
            top_p = 0.95,
            seed = None,
            repetition_penalty = 1.03,
        )
    )
]
```

<span style="color: red; font-weight: bold; font-size: 18px;">**跑不同并发数和请求频率的性能，在此处修改batch_size和request_rate**
</span>

### 4.3 修改ceval_gen_0_shot_cot_chat_prompt.py

```
cd /opt/package/benchmark/ais_bench/benchmark/configs/datasets/ceval/
vim ceval_gen_0_shot_cot_chat_prompt.py
```

只需要修改第89行的数据集路径，即修改为绝对路径，其他数据集以此类推。如果没有修改，在`/opt/package/benchmark`路径下执行才可以执行对应精度性能任务。

### 4.4 测试命令

**性能测试**：
gsm8k数据集case为20的性能测试命令

```
ais_bench --models vllm_api_stream_chat  --datasets gsm8k_gen_0_shot_cot_str_perf  --mode perf --summarizer default_perf --debug --num-prompts 20
```

精度测试：

```
# gsm8k数据集测试命令
ais\_bench --models vllm\_api\_general\_chat --datasets gsm8k\_gen\_0\_shot\_cot\_chat\_prompt
# mmlu数据集测试命令
ais\_bench --models vllm\_api\_general\_chat --datasets mmlu\_gen\_0\_shot\_cot\_chat\_prompt --merge-ds
# ceval数据集测试命令
ais\_bench --models vllm\_api\_general\_chat --datasets ceval\_gen\_0\_shot\_cot\_chat\_prompt --merge-ds
```