# README

- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)是目前Qwen系列中最强大的多模态大模型，这一代在各个方面都进行了全面升级：更优秀的文本理解和生成、更深的视觉感知和推理、扩展的上下文长度、增强的空间和视频动态理解能力，以及更强的代理交互能力。
- 此代码仓中实现了一套基于NPU硬件的Qwen3-VL Dense和MoE架构推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。
- 支持Qwen3-VL Dense和MoE架构模型的多模态推理

# 特性矩阵
- 此矩阵罗列了Qwen3-VL模型支持的特性

| 模型及参数量 | 模型架构 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | MindIE Service | 纯模型支持模态 | 服务化支持模态 | W8A8量化 | W8A8SC量化 | 并行解码 |
|-------------|----------------------------|-----------------------------|------|------|----------------|---------------|---------------|---------|------------|-------------|---------|
| Qwen3-VL-2B-Instruct | Dense | 支持world size 1,2,4,8(推荐1) | 支持world size 1,2,4,8(推荐1) | 是 | 是 | 是 | 文本、图片、视频 | 文本、图片、视频 |×|×|✓|
| Qwen3-VL-4B-Instruct | Dense | 支持world size 1,2,4,8(推荐1) | 支持world size 1,2,4,8(推荐1) | 是 | 是 | 是 | 文本、图片、视频 | 文本、图片、视频 |×|✓|✓|
| Qwen3-VL-8B-Instruct | Dense | 支持world size 1,2,4,8(推荐2) | 支持world size 1,2,4,8(推荐2) | 是 | 是 | 是 | 文本、图片、视频 | 文本、图片、视频 |✓|✓|✓|
| Qwen3-VL-32B-Instruct | Dense | 支持world size 2,4,8(推荐4) | 支持world size 2,4,8(推荐4) | 是 | 是 | 是 | 文本、图片、视频 | 文本、图片、视频 |✓|✓|✓|
| Qwen3-VL-30B-A3B-Instruct | MoE | 支持world size 2,4(推荐2) | 支持world size 2,4(推荐2) | 是 | 是 | 是 | 文本、图片、视频 | 文本、图片、视频 |✓|×|✓|

注意：
- Qwen3-VL-30B-A3B-Instruct模型的W8A8量化是混合量化（Attention:w8a8量化，MoE:w8a8 dynamic量化）

# 使用说明

## 路径变量解释

| 变量名      | 含义                                                                                                                                                         |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| working_dir | 加速库及模型库下载后放置的目录                                                                                                                               |
| llm_path    | 模型仓所在路径。若使用编译好的包，则路径为 `${working_dir}/MindIE-LLM/`；若使用gitcode下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径；qwen3vl的工作脚本所在路径为 `${llm_path}/examples/models/qwen3_vl`                                                                                          |
| weight_path | 模型权重路径                                                                     |
| image_path  | 图片所在路径                                                                     |
| trust_remote_code  | 是否信任本地可执行文件，默认为False，若设置该参数则为True                     |
| max_batch_size  | 最大bacth数                                                                  |
| max_input_length  | 多模态模型的最大embedding长度，                                             |
| max_output_length | 生成的最大token数                                                          |

- 注意：
max_input_length长度设置可参考模型权重路径下config.json里的max_position_embeddings参数值
## 权重

### 量化权重生成
**Dense架构模型**
- 参考[Qwen3-VL 量化案例](https://gitcode.com/Ascend/msit/blob/master/msmodelslim/example/multimodal_vlm/Qwen3-VL/README.md)

**MoE架构模型**
- 参考[Qwen3-VL-MoE 量化使用说明](https://gitcode.com/Ascend/msit/blob/master/msmodelslim/example/multimodal_vlm/Qwen3-VL-MoE/README.md)

注意：
1. 仅300I DUO平台支持稀疏压缩量化（W8A8SC）
2. 为保证模型精度和性能，在300I DUO平台使用的W8A8权重，需在800I A2平台量化完成后使用[deq_scale_cast.py](https://gitcode.com/Ascend/msit/blob/master/msmodelslim/example/deq_scale_cast.py)转换后使用
### 昇腾原生量化权重下载

也可以通过ModelScope 魔搭社区直接下载昇腾原生量化模型权重：

**300I DUO**

- [Qwen3-VL-4B-Instruct-w8a8s-310](https://modelscope.cn/models/Eco-Tech/Qwen3-VL-4B-Instruct-w8a8s-310)
- [Qwen3-VL-8B-Instruct-w8a8s-310](https://www.modelscope.cn/models/Eco-Tech/Qwen3-VL-8B-Instruct-w8a8s-310)
- [Qwen3-VL-32B-Instruct-w8a8s-310](https://www.modelscope.cn/models/Eco-Tech/Qwen3-VL-32B-Instruct-w8a8s-310)
- [Qwen3-VL-30B-A3B-Instruct-w8a8-QuaRot-310](https://www.modelscope.cn/models/Eco-Tech/Qwen3-VL-30B-A3B-Instruct-w8a8-QuaRot-310)

- 注意：  
稀疏压缩权重需根据实际使用TP数自主压缩，压缩命令：  
torchrun --nproc_per_node {TP数} -m examples.convert.model_slim.sparse_compressor --model_path {W8A8S量化权重路径} --save_directory {W8A8SC量化权重路径} --multiprocess_num 4  
权重压缩后，需手动将浮点权重路径下的chat_template.json，preprocessor_config.json，video_preprocessor_config.json三个文件拷贝至W8A8SC量化权重路径下。

**800I A2**

- [Qwen3-VL-8B-Instruct-w8a8-QuaRot](https://modelscope.cn/models/Eco-Tech/Qwen3-VL-8B-Instruct-w8a8-QuaRot)
- [Qwen3-VL-32B-Instruct-w8a8-QuaRot](https://modelscope.cn/models/Eco-Tech/Qwen3-VL-32B-Instruct-w8a8-QuaRot)
- [Qwen3-VL-30B-A3B-Instruct-w8a8-QuaRot](https://www.modelscope.cn/models/Eco-Tech/Qwen3-VL-30B-A3B-Instruct-w8a8-QuaRot)

## 推理

**基础环境配置**

1. Python其他第三方库依赖，参考[requirements_qwen3vl.txt](../../../requirements/models/requirements_qwen3vl.txt)
2. 参考[此README文件](../../../README.md)

### 对话测试

**运行Paged Attention纯模型推理脚本**

- 运行启动脚本
  - 在\${llm_path}目录下执行以下指令
    ```shell
    bash ${script_path}/run_pa.sh --trust_remote_code --model_path ${weight_path} --image_path ${image_path} ${max_batch_size} ${max_input_length} ${max_output_length}
    ```
- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1`
    - 指定当前机器上可用的逻辑NPU核心，多个核心间使用逗号相连
    - 核心ID查阅方式见[此README文件](../../README.md)的【启动脚本相关环境变量】章节
    - 各模型支持的核心数参考“特性矩阵”
  - `export MASTER_PORT=20030`
    - 设置卡间通信端口
    - 默认使用20030端口
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

## 服务化推理

- 打开Server配置文件

- 配置端口、硬件id、模型名称、和权重路径

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
"npuDeviceIds" : [[0,1]], # 芯片id，按需配置
...
"ModelDeployConfig":
{
"maxSeqLen" : 10240,
"maxInputTokenLen" : 10240,
"truncation" : false,
"ModelConfig" : [
{
"modelInstanceType": "Standard",
"modelName" : "qwen3vl", # 模型名称配置为qwen3vl
"modelWeightPath" : "/data_mm/weights/Qwen3-VL-8B-Instruct", # 自定义本地权重路径
"worldSize" : 2, # 并行卡数，按需配置
...
"npuMemSize" : 3, # 文本模型KV Cache内存分配，可自行调整，单位是GB
...
"trustRemoteCode" : false # 默认为false，若设为true，则信任本地代码，用户需自行承担风险
}
]
}
```

- 启动服务端进程

- 新建窗口，测试vLLM请求接口

```shell
curl 127.0.0.1:1040/generate -d '{
"prompt": [
{"type": "text", "text": "描述这张图片"}，
{
"type": "image_url",
"image_url": ${图片路径}
}
],
"max_tokens": 128,
"model": "qwen3vl"
}'
```
或
```shell
curl 127.0.0.1:1040/generate -d '{
"prompt": [
{"type": "text", "text": "描述这个视频"}，
{
"type": "video_url",
"video_url": ${视频路径}
}
],
"max_tokens": 128,
"model": "qwen3vl"
}'
```

- 测试OpenAI请求接口

```shell
curl 127.0.0.1:1040/v1/chat/completions -d ' {
"model": "qwen3vl",
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

## Aisbench精度测试
- 首先按照[服务化推理](#服务化推理)，启动服务端进程

- 参考[Aisbench/benchmark](https://github.com/AISBench/benchmark/)安装精度性能评测工具
- 参考[开源数据集](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/textvqa/README.md)下载Textvqa数据集
- 配置测试任务
```python
...
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="/data_mm/weights/Qwen3-VL-8B-Instruct", # 自定义本地权重路径
        model="qwen3vl", # 模型名称配置为qwen3vl
        stream=False,
        request_rate=0,
        retry=2,
        api_key="",
        host_ip="localhost", # 服务IP地址
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
ais_bench --models vllm_api_general_chat --datasets textvqa_gen_base64 --mode all --debug
```

## Aisbench性能测试
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