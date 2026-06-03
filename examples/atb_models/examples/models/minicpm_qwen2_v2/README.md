# README

- [MiniCPM-V-2_6](https://github.com/OpenBMB/MiniCPM-V) 是面向图文理解的端侧多模态大模型系列。该系列模型接受图像和文本输入，并提供高质量的文本输出.
- 此代码仓中实现了一套基于 NPU 硬件的 MiniCPM-V 推理模型。配合加速库使用，旨在 NPU 上获得极致的推理性能。

## 使用说明

### 特性矩阵

| 模型及参数量                     | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | 800I A2 BF16 | MindIE Service | 纯模型支持模态 | 服务化支持模态    |
|----------------------------|------------|---------------------------|------|------------|----------------|---------|------------|
| MiniCPM-V-2_6, 8B | 支持 TP 1、2、4、8      | 支持 TP 1、2           | ✅    | ✅          | ✅              | 文本、图片、视频   | 单轮对话/多轮对话 |

### 路径变量解释

| 变量名               | 含义                                                                                                                                                             |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| working_dir       | 加速库及模型库下载后放置的目录                                                                                                                                                |
| llm_path          | 模型仓所在路径。若使用镜像，则路径为 `/usr/local/Ascend/atb-models`；若使用 gitcode 下载的代码，则路径为 `{working_dir}/MindIE-LLM/examples/atb_models`（仅源码下载场景使用）                                          |
| script_path       | 脚本所在路径；minicpm_qwen2_v2 的工作脚本所在路径为 `${llm_path}/examples/models/minicpm_qwen2_v2`                                                                                               |
| weight_path       | 模型权重路径                                                                                                                                                         |
| trust_remote_code  | 是否信任本地的可执行文件：默认不执行，传入此参数，则信任                                                                                                                |
| image_or_video_path        | 图片或视频所在文件夹的路径。当前图片仅支持 ".jpg", ".png", ".jpeg", ".bmp" 四种格式。视频仅支持 ".mp4", ".wmv", ".avi" 三种格式                                                                                                                              |
| max_batch_size    | 最大 batch 数                                                                                                                                                       |
| max_input_length  | 多模态模型的最大 embedding 长度。 |
| max_output_length | 生成的最大 token 数                                                                                                                                                    |

### 推理

**权重下载**

- [MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6/tree/main)

**模型文件拷贝**

将权重目录中的 resampler.py 拷贝到 ${llm_path}/examples/atb_models/atb_llm/models/minicpm_qwen2_v2 目录下

**基础环境变量**

- 1.Toolkit, MindIE/ATB, ATB-SPEED 等，参考[此 README 文件](../../../README.md)
- 2.Python 其他第三方库依赖，参考 [requirements_minicpm_qwen2_v2.txt](../../../requirements/models/requirements_minicpm_qwen2_v2.txt)

  ```shell
  pip install -r ${llm_path}/requirements/models/requirements_minicpm_qwen2_v2.txt
  ```

#### 对话测试

- 运行启动脚本
  - 在 ${llm_path} 目录下执行以下命令

    ```shell
    bash ${script_path}/run_pa.sh --run --trust_remote_code ${weight_path} ${image_path}
    ```

- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0`

  - 以下环境变量与性能和内存优化相关，通常情况下无需修改

    ```shell
    export INF_NAN_MODE_ENABLE=0
    export ATB_OPERATION_EXECUTE_ASYNC=1
    export TASK_QUEUE_ENABLE=1
    export ATB_CONVERT_NCHW_TO_ND=1
    export ATB_CONTEXT_WORKSPACE_SIZE=0
    ```

## 服务化推理

- 打开配置文件

```shell
vim /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
```

- 更改配置文件

```json
{
...
"ServerConfig" :
{
...
"port" : 1040, #自定义
"managementPort" : 1041, #自定义
"metricsPort" : 1042, #自定义
...
"httpsEnabled" : false,
...
},

"BackendConfig": {
...
"npuDeviceIds" : [[0,1,2,3,4,5,6,7]],
...
"ModelDeployConfig":
{
"maxSeqLen" : 16384,
"maxInputTokenLen" : 16384,
"truncation" : false,
"ModelConfig" : [
{
"modelInstanceType": "Standard",
"modelName" : "minicpm_qwen2_v2", # 为了方便使用 benchmark 测试，modelname 建议使用 internvl
"modelWeightPath" : "/data_mm/weights/MiniCPM-V-2_6",
"worldSize" : 8,
...
"npuMemSize" : 1, #kvcache 分配，可自行调整，单位是 GB，切勿设置为-1，需要给 vit 预留显存空间。32GB 机器建议设为 1, 64GB 机器可以设为 8。
...
"trustRemoteCode" : false #默认为 false，若设为 true，则信任本地代码，用户需自行承担风险
}
]
},
"ScheduleConfig" :
{
...
"maxPrefillTokens" : 50000,
"maxIterTimes": 4096,
...
}
}
}
```

- 拉起服务化

```shell
cd /usr/local/Ascend/mindie/latest/mindie-service/bin
./mindieservice_daemon
```

- 另起一个新的容器端口，测试 VLLM 接口

```shell
curl 127.0.0.1:1040/generate -d '{
"prompt": [
{
"type": "image_url",
"image_url": ${图片路径}
},
{"type": "text", "text": "Explain the details in the image."}
],
"max_tokens": 512,
"stream": false,
"do_sample":true,
"repetition_penalty": 1.00,
"temperature": 0.01,
"top_p": 0.001,
"top_k": 1,
"model": "minicpm_qwen2_v2"
}'
```

- 另起一个新的容器端口，测试 OpenAI 接口

```shell
curl 127.0.0.1:1040/v1/chat/completions -d ' {
"model": "minicpm_qwen2_v2",
"messages": [{
"role": "user",
"content": [
{"type": "image_url", "image_url": ${图片路径}},
{"type": "text", "text": "Explain the details in the image."}
]
}],
"max_tokens": 512,
"do_sample": true,
"repetition_penalty": 1.00,
"temperature": 0.01,
"top_p": 0.001,
"top_k": 1
}'
```

## 精度测试

- 首先按照[服务化推理](#服务化推理)，拉起服务化

- 参考 [AISBench](https://github.com/AISBench/benchmark/) 安装精度性能评测工具
- 数据准备
  - 数据集下载 [Eval_QA](https://huggingface.co/datasets/maoxx241/videobench_subset) && [Video-Bench](https://huggingface.co/datasets/LanguageBind/Video-Bench/tree/main)
  - 将 `Eval_QA/` 目录下各 json 文件中的 `vid_path` 属性值改为相应视频的绝对路径

  ```json
  ...
  "v_C7yd6yEkxXE_4": {
    "vid_path": "/data_mm/Eval_video/ActivityNet/v_C7yd6yEkxXE.mp4"
  }
  ...
  ```

- 使用 `videobench` 数据集任务进行精度测试
- 配置测试任务 `ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py`

```python
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="/data_mm/weights/MiniCPM-V-2_6", # 自定义本地权重路径
        model="minicpm_qwen2_v2", # 模型名称配置为 minicpm_qwen2_v2
        stream=False,
        request_rate=0,
        retry=2,
        api_key="",
        host_ip="localhost", # 服务 IP 地址
        host_port=1040, # 服务业务面端口号，与服务化推理配置保持一致
        url="",
        max_out_len=16384,
        batch_size=1,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.01,
            ignore_eos=False
        )
    )
]
```

执行命令开始精度测试

```shell
ais_bench --models vllm_api_general_chat --datasets videobench --mode all --debug
```

## 性能测试

使用 [AISBench](https://github.com/AISBench/benchmark) 工具进行性能测试。

**1、配置 `vllm_api_stream_chat.py`**

```python
from ais_bench.benchmark.models import VLLMCustomAPIChatStream

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="",
        model="minicpm_qwen2_v2", # 与服务化 config.json 中的 modelName 一致
        request_rate=0,
        retry=2,
        host_ip="127.0.0.1", # 推理服务 IP（多机场景为主节点 IP）
        host_port=1040, # 服务化 config.json 中配置的 port
        max_out_len=512, # 最大输出 token 数
        batch_size=16, # 最大并发数
        generation_kwargs=dict(
            temperature=0,
            ignore_eos=True, # 测试定长输出时需设为 True
        )
    )
]
```

**2、执行性能测试**

```shell
ais_bench --models vllm_api_stream_chat --datasets textvqa_gen --mode perf --summarizer default_perf --debug
```

关键性能指标：

- **TTFT**（Time To First Token）：首 token 延迟，发送请求到收到第一个输出 token 的时间。
- **TPOT**（Time Per Output Token）：每 token 延迟，decode 阶段平均生成一个 token 所需时间。
- **Prefill Token Throughput**：prefill 阶段每秒处理的 token 数。
- **Output Token Throughput**：decode 吞吐，每秒生成的输出 token 数。
