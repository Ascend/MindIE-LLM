# README

- [GLM-4.1V-9B-Thinking](https://github.com/zai-org/GLM-V)，是智谱 AI 推出的最新一代预训练模型 GLM-4.1V 系列中的开源多模态视觉语言大模型。GLM-4.1V-9B-Thinking 引入思考范式，通过课程采样强化学习 RLCS（Reinforcement Learning with Curriculum Sampling）全面提升模型能力， 达到 10B 参数级别的视觉语言模型的最强性能。
- 此代码仓中实现了一套基于 NPU 硬件的 GLM-4.1V-9B-Thinking 推理模型。配合加速库使用，旨在 NPU 上获得极致的推理性能。
- 支持 GLM-4.1V-9B-Thinking 模型的多模态推理

## 特性矩阵

- 此矩阵罗列了 GLM-4.1V-9B-Thinking 模型支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | MindIE Service | 纯模型支持模态 | 服务化支持模态 | W8A8 量化 | W8A8SC 量化 | Micro batch | 并行解码 |
|-------------|----------------------------|-----------------------------|------|------|----------------|---------------|---------------|---------|------------|-------------|---------|
| GLM-4.1V-9B-Thinking | 支持 world size 1,2 | 支持 world size 1,2 | ✅ | ✅ | ✅ | 文本、图片、视频 | 文本、图片、视频 |✅|✅|✅|✅|

须知：服务化请求支持多张图片输入，但不支持一个请求内图片视频混合输入。

## 使用说明

### 路径变量解释

| 变量名      | 含义                                                                                                                                                         |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| working_dir | 加速库及模型库下载后放置的目录（仅源码下载场景使用）                                                                                               |
| llm_path    | 模型仓所在路径。若使用镜像，则路径为 `/usr/local/Ascend/atb-models`；若使用 gitcode 下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径；glm4v 的工作脚本所在路径为 `${llm_path}/examples/models/glm41v`                                                                                          |
| weight_path | 模型权重路径                                                                     |
| image_path  | 图片所在路径                                                                     |
| trust_remote_code  | 是否信任本地可执行文件，默认为 False，若设置该参数则为 True                     |
| max_batch_size  | 最大 batch 数                                                                |
| max_input_length  | 多模态模型的最大 embedding 长度                                              |
| max_output_length | 生成的最大 token 数                                                          |

-注意：
max_input_length 长度设置可参考模型权重路径下 config.json 里的 max_position_embeddings 参数值

### 权重

**权重下载**

- [GLM-4.1V-9B-Thinking](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking)
-注意：
开源权重于 2025-10-25 日更新了 config.json 格式，需使用 2025-10-25 日后的权重版本。

**权重量化**

- 参考 [msmodelslim](https://gitcode.com/Ascend/msmodelslim/blob/26.1.0/example/multimodal_vlm/GLM-4.1V/README.md)
- 800I A2 支持 w8a8 量化，命令如下
    python quant_glm4_1v.py --model_path {浮点权重路径} --save_directory {量化权重保存路径} --calib_images {校准图片路径} --w_bit 8 --a_bit 8 --device_type npu --anti_method m2 --torch_dtype bf16 --trust_remote_code True
- 300I DUO 支持 w8a8sc 量化，命令如下

  - 昇腾原生量化权重下载 或 生成量化权重 二选一：

    - 可以通过 ModelScope 魔搭社区直接下载昇腾原生量化模型权重：

    [GLM-4.1V-9B-Thinking-w8a8s-310](https://www.modelscope.cn/models/Eco-Tech/GLM-4.1V-9B-Thinking-w8a8s-310)

    - 生成量化权重：
    python quant_glm41v.py --model_path {浮点权重路径} --save_directory {W8A8S 量化权重路径} --calib_images {校准集图片路径} --w_bit 4 --a_bit 8 --device_type npu --anti_method m2 --is_lowbit True --fraction 0.01 --use_sigma True --torch_dtype fp16 --trust_remote_code True
  - 量化权重切分及压缩：
  torchrun --nproc_per_node {TP 数} -m examples.convert.model_slim.sparse_compressor --model_path {W8A8S 量化权重路径} --save_directory {W8A8SC 量化权重路径} --multiprocess_num 4
  权重压缩后，需手动将浮点权重路径下的 chat_template.jinja，preprocessor_config.json，video_preprocessor_config.json 三个文件拷贝至 W8A8SC 量化权重路径下。

**基础环境变量**

1. Python 其他第三方库依赖，参考 [requirements_glm4v.txt](../../../requirements/models/requirements_glm4.1v.txt)
2. 参考[此 README 文件](../../../README.md)

### 推理

#### 对话测试

**运行 Paged Attention 纯模型推理脚本**

- 运行启动脚本
  - 在\${llm_path} 目录下执行以下指令

    ```shell
    bash ${script_path}/run_pa.sh --trust_remote_code --model_path ${weight_path} --image_path ${image_path} ${max_batch_size} ${max_input_length} ${max_output_length}
    ```

- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1`
    - 指定当前机器上可用的逻辑 NPU 核心，多个核心间使用逗号相连
    - 核心 ID 查阅方式见[此 README 文件](../../README.md)的【启动脚本相关环境变量】章节
    - 各模型支持的核心数参考“特性矩阵”
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
    export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
    export ATB_CONTEXT_WORKSPACE_SIZE=0
    ```

### 服务化推理

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
"httpsEnabled" : false,
...
},

"BackendConfig": {
...
"npuDeviceIds" : [[0,1]], # 芯片 id，按需配置
...
"ModelDeployConfig":
{
"maxSeqLen" : 10240,
"maxInputTokenLen" : 10240,
"truncation" : false,
"ModelConfig" : [
{
"modelInstanceType": "Standard",
"modelName" : "glm4v", # 模型名称配置为 glm4v
"modelWeightPath" : "/data_mm/weights/GLM-4.1V-9B-Thinking", # 自定义本地权重路径
"worldSize" : 2, # 并行卡数，按需配置
...
"npuMemSize" : 15, # 文本模型 KV Cache 内存分配，可自行调整，单位是 GB
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

或

```shell
curl 127.0.0.1:1040/generate -d '{
"prompt": [
{"type": "text", "text": "描述这个视频"},
{
"type": "video_url",
"video_url": ${视频路径}
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

### AISBench 精度测试

- 首先按照[服务化推理](#服务化推理)，启动服务端进程

- 参考 [AISBench/benchmark](https://github.com/AISBench/benchmark/) 安装精度性能评测工具
- 参考[开源数据集](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/textvqa/README.md)下载 Textvqa 数据集
- 配置测试任务

```python
...
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="/data_mm/weights/GLM-4.1V-9B-Thinking", # 自定义本地权重路径
        model="glm4v", # 模型名称配置为 glm4v
        stream=False,
        request_rate=0,
        retry=2,
        api_key="",
        host_ip="localhost", # 服务 IP 地址
        host_port=8080, # 服务业务面端口号
        url="",
        max_out_len=2048,
        batch_size=32,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature = 0.0
        )
    )
]
```

执行命令开始精度测试

```shell
ais_bench --models vllm_api_general_chat --datasets glm4v_textvqa_gen_base64 --mode all --debug
```

### AISBench 性能测试

- 首先按照[服务化推理](#服务化推理)，启动服务端进程
- 配置测试任务(mm_custom_gen)

```python
...
mm_custom_reader_cfg = dict(
    input_columns=['question', 'image'],
    output_column='answer'
)


mm_custom_infer_cfg = dict(
    prompt_template=dict(
        type=MMPromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt_mm={
                    "text": {"type": "text", "text": "{question}"},
                    "image": {"type": "image_url", "image_url": {"url": "file://{image}"}}
                })
            ]
            )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

...
mm_custom_datasets = [
    dict(
        abbr='mm_custom',
        type=MMCustomDataset,
        path='/data_mm/dataset.jsonl', # 自定义本地数据集路径
        mm_type="path",
        num_frames=5,
        reader_cfg=mm_custom_reader_cfg,
        infer_cfg=mm_custom_infer_cfg,
        eval_cfg=mm_custom_eval_cfg
    )
]
```

执行命令开始性能测试

```shell
ais_bench --models vllm_api_stream_chat --datasets mm_custom_gen --mode perf --debug
```
