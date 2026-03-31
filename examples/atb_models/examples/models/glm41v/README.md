# README

- [GLM-4.1V-9B-Thinking](https://github.com/zai-org/GLM-V)，是智谱AI推出的最新一代预训练模型GLM-4.1V系列中的开源多模态视觉语言大模型。GLM-4.1V-9B-Thinking引入思考范式，通过课程采样强化学习 RLCS（Reinforcement Learning with Curriculum Sampling）全面提升模型能力， 达到 10B 参数级别的视觉语言模型的最强性能。
- 此代码仓中实现了一套基于NPU硬件的GLM-4.1V-9B-Thinking推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。
- 支持GLM-4.1V-9B-Thinking模型的多模态推理

# 特性矩阵

- 此矩阵罗列了GLM-4.1V-9B-Thinking模型支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | MindIE Service | 纯模型支持模态 | 服务化支持模态 | W8A8量化 | W8A8SC量化 | Micro batch | 并行解码 |
|-------------|----------------------------|-----------------------------|------|------|----------------|---------------|---------------|---------|------------|-------------|---------|
| GLM-4.1V-9B-Thinking | 支持world size 1,2 | 支持world size 1,2 | 是 | 是 | 是 | 文本、图片、视频 | 文本、图片、视频 |✓|✓|✓|✓|

须知：服务化请求支持多张图片输入，但不支持一个请求内图片视频混合输入。

# 使用说明

## 路径变量解释

| 变量名      | 含义                                                                                                                                                         |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| working_dir | 加速库及模型库下载后放置的目录                                                                                                                               |
| llm_path    | 模型仓所在路径。若使用编译好的包，则路径为 `${working_dir}/MindIE-LLM/`；若使用gitcode下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径；glm4v的工作脚本所在路径为 `${llm_path}/examples/models/glm41v`                                                                                          |
| weight_path | 模型权重路径                                                                     |
| image_path  | 图片所在路径                                                                     |
| trust_remote_code  | 是否信任本地可执行文件，默认为False，若设置该参数则为True                     |
| max_batch_size  | 最大bacth数                                                                  |
| max_input_length  | 多模态模型的最大embedding长度，                                             |
| max_output_length | 生成的最大token数                                                          |

-注意：
max_input_length长度设置可参考模型权重路径下config.json里的max_position_embeddings参数值

## 权重

**权重下载**

- [GLM-4.1V-9B-Thinking](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking)
-注意：
开源权重于2025-10-25日更新了config.json格式，需使用2025-10-25日后的权重版本。

**权重量化**

- 参考[msit](https://gitcode.com/Ascend/msit/blob/master/msmodelslim/example/multimodal_vlm/GLM-4.1V/README.md)
- 800I A2支持w8a8量化，命令如下  
    python quant_glm4_1v.py --model_path {浮点权重路径} --save_directory {量化权重保存路径} --calib_images {校准图片路径} --w_bit 8 --a_bit 8 --device_type npu --anti_method m2 --torch_dtype bf16 --trust_remote_code True  
- 300I DUO支持w8a8sc量化，命令如下

  - 昇腾原生量化权重下载 或 生成量化权重 二选一：

    - 可以通过ModelScope魔搭社区直接下载昇腾原生量化模型权重：

    [GLM-4.1V-9B-Thinking-w8a8s-310](https://www.modelscope.cn/models/Eco-Tech/GLM-4.1V-9B-Thinking-w8a8s-310)

    - 生成量化权重：  
    python quant_glm41v.py --model_path {浮点权重路径} --save_directory {W8A8S量化权重路径} --calib_images {校准集图片路径} --w_bit 4 --a_bit 8 --device_type npu --anti_method m2 --is_lowbit True --fraction 0.01 --use_sigma True --torch_dtype fp16 --trust_remote_code True
  - 量化权重切分及压缩：  
  torchrun --nproc_per_node {TP数} -m examples.convert.model_slim.sparse_compressor --model_path {W8A8S量化权重路径} --save_directory {W8A8SC量化权重路径} --multiprocess_num 4  
  权重压缩后，需手动将浮点权重路径下的chat_template.jinja，preprocessor_config.json，video_preprocessor_config.json三个文件拷贝至W8A8SC量化权重路径下。

**基础环境变量**

1. Python其他第三方库依赖，参考[requirements_glm4v.txt](../../../requirements/models/requirements_glm4.1v.txt)
2. 参考[此README文件](../../../README.md)

## 推理

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

```shell
vim /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
```

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
"modelName" : "glm4v", # 模型名称配置为glm4v
"modelWeightPath" : "/data_mm/weights/GLM-4.1V-9B-Thinking", # 自定义本地权重路径
"worldSize" : 2, # 并行卡数，按需配置
...
"npuMemSize" : 15, # 文本模型KV Cache内存分配，可自行调整，单位是GB
...
"trustRemoteCode" : false # 默认为false，若设为true，则信任本地代码，用户需自行承担风险
}
]
}
```

- 启动服务端进程

```shell
cd /usr/local/Ascend/mindie/latest/mindie-service/bin
./mindieservice_daemon
```

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
"model": "glm4v"
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
"model": "glm4v"
}'
```

- 测试OpenAI请求接口

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
        path="/data_mm/weights/GLM-4.1V-9B-Thinking", # 自定义本地权重路径
        model="glm4v", # 模型名称配置为glm4v
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
ais_bench --models vllm_api_general_chat --datasets glm4v_textvqa_gen_base64 --mode all --debug
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
