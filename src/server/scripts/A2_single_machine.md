# A2单机PD混部一键部署
以下配置以cann B080, deepseekR1, w4a8为例，NPU_MEMORY_FRACTION=0.92适配deepseekR1, w8a8的环境，其他参数，权重路径需要根据模型实际情况调整
## 1. 配置环境变量
需要修改libjemalloc.so路径后
```bash
source A2_single_machine.sh
```

### 1.1 环境变量说明
- NPU_MEMORY_FRACTION：参数为显存比例因子，默认参数为0.92，高并发场景下建议 <=0.92。
    - 建议配置方案：建议将该值设置为可拉起服务的最小值。具体方法是，按照默认配置启动服务，若无法拉起服务，则上调参数至可拉起为止；若拉起服务成功，则下调该参数至刚好拉起服务为止。总之，在服务能正常拉起的前提下，更低的值可以保障更高的服务系统稳定性。

## 2. 配置性能测试mindie_service/conf/config.json文件
打开mindie_service/conf/config.json文件，修改以下参数：

| origin | change  |
|---|---|
|  "httpsEnabled" : true, | "httpsEnabled" : false,  |
| "interCommTLSEnabled" : true,  | "interCommTLSEnabled" : false,  |
| "tokenTimeout" : 600,  | "tokenTimeout" : 3600,  |
| "e2eTimeout" : 600,  | "e2eTimeout" : 65535,  |
|  "npuDeviceIds" : [[0,1,2,3]], | "npuDeviceIds" : [[0,1,2,3,4,5,6,7]],  |
| "multiNodesInferEnabled" : false,  | "multiNodesInferEnabled" : true,  |
| "interNodeTLSEnabled" : true,  | "interNodeTLSEnabled" : false,  |
| "maxSeqLen" : 2560,  | "maxSeqLen" : 28672,  |
| "maxInputTokenLen" : 2560,  | "maxInputTokenLen" : 28672,  |
|  "modelName" : "llama_65b", | "modelName" : "dsr1",  |
| "modelWeightPath" : "/data/atb_testdata/weights/llama1-65b-safetensors",  | "modelWeightPath" : "/mnt/mindie_data/weight/Deepseek_w4a8_0625",  |
| "worldSize" : 4,  | "worldSize" : 8,  |
| "maxPrefillTokens" : 8192,   |  "maxPrefillTokens" : 28672,  |
| "maxIterTimes" : 512,   |  "maxIterTimes" : 28672,  |

在"async_scheduler_wait_time": 120,后面加上
```bash
"moe_ep": 1,
"moe_tp": 8,
"sp": 1,
"tp": 8,
"dp": 1,
"ignore_eos": true,
"plugin_params": "{\"plugin_type\":\"mtp\",\"num_speculative_tokens\": 1}",
"models": {
    "deepseekv2":{
        "enable_mlapo_prefetch": true,
        "kv_cache_options": {"enable_nz": true}
    }
}
```
## 3.配置性能测试脚本ais_bench/benchmark/configs/modles/vllm_api/vllm_api_stream_chat.py
需要修改host_ip后，直接复制
```bash
from ais_bench.benchmark.models import VLLMCustomAPIChatStream

models=[
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="/mnt/mindie_data/weight/Deepseek_w4a8_0625",
        model="dsr1",
        request_rate=0,
        retry=2,
        host_ip="host_ip",
        host_port=1025,
        max_out_len=1500,
        batch_size=25,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.6,
            ignore_eos=True,
        )
    )
]
```
## 4. 配置精度测试mindie_service/conf/config.json文件
只需将性能测试config.json里的"ignore_eos": true,改成"ignore_eos": false,

## 5.配置精度测试脚本ais_bench/benchmark/configs/modles/vllm_api/vllm_api_general_chat.py
需要修改host_ip后，直接复制
```bash
from ais_bench.benchmark.models import VLLMCustomAPIChat

models=[
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="/mnt/mindie_data/weight/Deepseek_w4a8_0625",
        model="dsr1",
        request_rate=0,
        retry=2,
        host_ip="host_ip",
        host_port=1025,
        max_out_len=24576,
        batch_size=2,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.6,
        )
    )
]
```
## 6.性能测试指令： 
```bash
ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_0_shot_cot_str_perf --debug --summarizer default_perf --mode perf --num-prompts 224
```
## 7.精度测试指令： 
```bash
ais_bench --models vllm_api_general_chat --datasets aime2024_gen_0_shot_chat_prompt --summarizer example
```