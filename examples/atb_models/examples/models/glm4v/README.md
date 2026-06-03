# README

- [GLM-4v-9B](https://github.com/THUDM/GLM-4)，是智谱 AI 推出的最新一代预训练模型 GLM-4 系列中的开源多模态版本。GLM-4v-9B 具备 1120*1120 高分辨率下的中英双语多轮对话能力，在中英文综合能力、感知推理、文字识别、图表理解等多方面多模态评测中，GLM-4v-9B 表现出超越 GPT-4-turbo-2024-04-09、Gemini 1.0 Pro、Qwen-VL-Max 和 Claude 3 Opus 的卓越性能。
- 此代码仓中实现了一套基于 NPU 硬件的 GLM-4v-9B 推理模型。配合加速库使用，旨在 NPU 上获得极致的推理性能。
- 支持 GLM-4v-9B 模型的多模态推理。

## 特性矩阵

此矩阵罗列了 GLM-4v-9B 模型支持的特性。

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | MindIE Service | 纯模型支持模态 | 服务化支持模态 |
|-------------|----------------------------|-----------------------------|------|------|----------------|-----------------|---------|
| GLM-4v-9B | 支持 world size 1,2,4,8 | ❌ | ✅ | ✅ | ✅ | 文本、图片 | 文本、图片 |

> 须知：服务化请求仅支持单张图片输入。

## 使用说明

### 路径变量解释

| 变量名      | 含义                                                                                                                                                         |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| working_dir | 加速库及模型库下载后放置的目录（仅源码下载场景使用）                                                                                                 |
| llm_path    | 模型仓所在路径。若使用镜像，则路径为 `/usr/local/Ascend/atb-models`；若使用 gitcode 下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径；glm4v 的工作脚本所在路径为 `${llm_path}/examples/models/glm4v`                                                                          |
| weight_path | 模型权重路径                                                                                                                                         |
| image_path  | 图片所在路径                                                                                                                                         |
| trust_remote_code  | 是否信任本地可执行文件，默认为 False，若设置该参数则为 True                                                                                          |
| max_batch_size  | 最大 batch 数                                                                                                                                        |
| max_input_length  | 多模态模型的最大 embedding 长度                                                                                                                      |
| max_output_length | 生成的最大 token 数                                                                                                                                  |

> **注意**：max_input_length 长度设置可参考模型权重路径下 config.json 里的 max_position_embeddings 参数值。

### 权重

**权重下载**

- [GLM-4v-9B](https://huggingface.co/THUDM/glm-4v-9b/tree/main)

**基础环境变量**

1. Python 其他第三方库依赖，参考 [requirements_glm4v.txt](../../../requirements/models/requirements_glm4v.txt)。
2. 参考[此 README 文件](../../../README.md)。

> **注意**：保证先后顺序，否则 glm4v 的其余三方依赖会重新安装 torch，导致出现别的错误。

### 推理

#### 对话测试

**运行 Paged Attention FP16**

- 运行启动脚本
  - 在 `${llm_path}` 目录下执行以下指令

    ```shell
    bash ${script_path}/run_pa.sh --run --trust_remote_code ${weight_path} ${image_path} ${max_batch_size} ${max_input_length} ${max_output_length}
    ```

- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
    - 指定当前机器上可用的逻辑 NPU 核心，多个核心间使用逗号相连
    - 核心 ID 查阅方式见[此 README 文件](../../README.md)的【启动脚本相关环境变量】章节
    - 各模型支持的核心数参考"特性矩阵"
  - `export MASTER_PORT=20030`
    - 设置卡间通信端口
    - 默认使用 20030 端口
    - 目的是为了避免同一台机器同时运行多个多卡模型时出现通信冲突
    - 设置时端口建议范围为：20000-20050
  - 以下环境变量与性能和内存优化相关，通常情况下无需修改

    ```shell
    export INF_NAN_MODE_ENABLE=0
    export ATB_OPERATION_EXECUTE_ASYNC=1
    export TASK_QUEUE_ENABLE=1
    export ATB_CONVERT_NCHW_TO_ND=1
    export ATB_CONTEXT_WORKSPACE_SIZE=0
    ```

## 服务化推理

- 打开 Server 配置文件

```shell
vim /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
```

- 配置端口、硬件 id、模型名称、和权重路径

```json
{
...
"ServerConfig" :
{
...
"port" : 1040, # 自定义
"managementPort" : 1041, # 自定义
"metricsPort" : 1042, # 自定义
...
"httpsEnabled" : false,
...
},

"BackendConfig": {
...
"npuDeviceIds" : [[0,1]], # 芯片 id，按需配置
...
"ModelDeployConfig":
{
"maxSeqLen" : 16384,
"maxInputTokenLen" : 16384,
"truncation" : false,
"ModelConfig" : [
{
"modelInstanceType": "Standard",
"modelName" : "glm4v", # 模型名称配置为 glm4v
"modelWeightPath" : "/data_mm/weights/glm-4v-9b", # 自定义本地权重路径
"worldSize" : 2, # 并行卡数，按需配置
...
"npuMemSize" : -1, # 文本模型 KV Cache 内存分配，可自行调整，单位是 GB
...
"trustRemoteCode" : false # 默认为 false，若设为 true，则信任本地代码，用户需自行承担风险
}
]
}
```

- 启动服务端进程

```shell
cd /usr/local/Ascend/mindie/latest/mindie-service/bin
./mindieservice_daemon
```

- 新建窗口，测试 vLLM 请求接口

```shell
curl 127.0.0.1:1040/generate -d '{
"prompt": [
{"type": "text", "text": "描述这张图片"},
{
"type": "image_url",
"image_url": ${图片路径}
}
],
"max_tokens": 128,
"model": "glm4v"
}'
```

- 测试 OpenAI 请求接口

```shell
curl 127.0.0.1:1040/v1/chat/completions -d ' {
"model": "glm4v",
"messages": [{
"role": "user",
"content": [
{"type": "image_url", "image_url": ${图片路径}},
{"type": "text", "text": "描述这张图片"}
]
}],
"max_tokens": 128,
"temperature": 1.0,
"top_p": 0.5,
"stream": false,
"repetition_penalty": 1.0,
"top_k": 10,
"do_sample": false
}'
```

## AISBench 精度测试

- 首先按照[服务化推理](#服务化推理)，启动服务端进程

- 参考 [AISBench/benchmark](https://github.com/AISBench/benchmark/) 安装精度性能评测工具
- 参考[开源数据集](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/textvqa/README.md)下载 TextVQA 数据集
- 精度测试使用 `glm4v_textvqa_gen_base64` 数据集任务
- 配置测试任务 `ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py`

```python
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-stream-chat',
        path="/data_mm/weights/glm-4v-9b", # 自定义本地权重路径
        model="glm4v", # 模型名称配置为 glm4v
        stream=True,
        request_rate=0,
        retry=2,
        api_key="",
        host_ip="localhost", # 服务 IP 地址
        host_port=1040, # 服务业务面端口号，与服务化推理配置保持一致
        url="",
        max_out_len=16384,
        batch_size=32,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.0,
            ignore_eos=False
        )
    )
]
```

执行命令开始精度测试

```shell
ais_bench --models vllm_api_stream_chat --datasets glm4v_textvqa_gen_base64 --mode all --debug
```
