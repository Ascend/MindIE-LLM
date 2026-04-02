# README

- [GLM-4v-9b](https://github.com/THUDM/GLM-4)，是智谱AI推出的最新一代预训练模型GLM-4系列中的开源多模态版本。GLM-4v-9B具备1120*1120高分辨率下的中英双语多轮对话能力，在中英文综合能力、感知推理、文字识别、图表理解等多方面多模态评测中，GLM-4v-9B表现出超越GPT-4-turbo-2024-04-09、Gemini 1.0 Pro、Qwen-VL-Max和Claude 3 Opus的卓越性能。
- 此代码仓中实现了一套基于NPU硬件的GLM-4v-9B推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。
- 支持GLM-4v-9B模型的多模态推理

# 特性矩阵
- 此矩阵罗列了GLM-4v-9b模型支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | MindIE Service | 纯模型支持模态 | 服务化支持模态 | 
|-------------|----------------------------|-----------------------------|------|----------------------|-----------------|-----------------|---------|
| GLM-4v-9B    | 支持world size 1,2,4,8     | 不支持           | 是   | 是                   | 是              | 文本、图片              | 文本、图片  |

须知：服务化请求仅支持单张图片输入。

# 使用说明

## 路径变量解释

| 变量名      | 含义                                                                                                                                                         |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| working_dir | 加速库及模型库下载后放置的目录                                                                                                                               |
| llm_path    | 模型仓所在路径。若使用编译好的包，则路径为 `${working_dir}/MindIE-LLM/`；若使用gitcode下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径；glm4v的工作脚本所在路径为 `${llm_path}/examples/models/glm4v`                                                                                          |
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

- [GLM-4v-9B](https://huggingface.co/THUDM/glm-4v-9b/tree/main)


**基础环境变量**

1. Python其他第三方库依赖，参考[requirements_glm4v.txt](../../../requirements/models/requirements_glm4v.txt)
2. 参考[此README文件](../../../README.md)
-注意：保证先后顺序，否则glm4v的其余三方依赖会重新安装torch，导致出现别的错误

## 推理

### 对话测试

**运行Paged Attention FP16**

- 运行启动脚本
  - 在\${llm_path}目录下执行以下指令
    ```shell
    bash ${script_path}/run_pa.sh --run --trust_remote_code ${weight_path} ${image_path} ${max_batch_size} ${max_input_length} ${max_output_length}
    ```
- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
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
"port" : 1040, # 自定义
"managementPort" : 1041, # 自定义
"metricsPort" : 1042, # 自定义
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
"maxSeqLen" : 16384,
"maxInputTokenLen" : 16384,
"truncation" : false,
"ModelConfig" : [
{
"modelInstanceType": "Standard",
"modelName" : "glm4v", # 模型名称配置为glm4v
"modelWeightPath" : "/data_mm/weights/glm-4v-9b", # 自定义本地权重路径
"worldSize" : 2, # 并行卡数，按需配置
...
"npuMemSize" : -1, # 文本模型KV Cache内存分配，可自行调整，单位是GB
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

## ModelTest精度测试
使用ModelTest测试下游精度数据集TextVQA
- 数据准备
    - 数据集下载 [textvqa](https://huggingface.co/datasets/maoxx241/textvqa_subset)
    - 保证textvqa_val.jsonl和textvqa_val_annotations.json在同一目录下
    - 将textvqa_val.jsonl文件中所有"image"属性的值改为相应图片的绝对路径
  ```json
  ...
  {
    "image": "/data/textvqa/train_images/003a8ae2ef43b901.jpg",
    "question": "what is the brand of this camera?",
    "question_id": 34602,
    "answer": "dakota"
  }
  ...
  ```
- 设置环境变量
  ```shell
  source /usr/local/Ascend/cann/set_env.sh
  source /usr/local/Ascend/nnal/atb/set_env.sh 
  source ${llm_path}/set_env.sh 
  ```
- 进入以下目录 `${llm_path}/tests/modeltest`
  ```shell
  cd ${llm_path}/tests/modeltest
  ```
- 安装modeltest及其三方依赖
  ```shell
  pip install --upgrade pip
  pip install -e .
  pip install tabulate termcolor 
  ```
- 将 `modeltest/config/model/glm4v.yaml` 中的model_path的值修改为模型权重的绝对路径
  ```yaml
  model_path: /data_mm/weights/glm-4v-9b
  ```
- 将 `modeltest/config/task/textvqa.yaml` 中的model_path修改为textvqa_val.jsonl文件的绝对路径
  ```yaml
  local_dataset_path: /data_mm/datasets/textvqa_val/textvqa_val.jsonl
  ```
- 设置可见卡数，修改 `scripts/mm_run.sh` 文件中的ASCEND_RT_VISIBLE_DEVICES。依需求设置单卡或多卡可见。
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1
  ```
- 运行测试命令
  ```shell
  bash scripts/mm_run.sh textvqa glm4v
  ```
- 测试结果保存于以下路径。其下的results/..(一系列文件夹嵌套)/\*\_result.csv中存放着modeltest的测试结果。debug/..(一系列文件夹嵌套)/output\_\*.txt中存储着每一条数据的运行结果，第一项为output文本，第二项为输入infer函数的第一个参数的值，即模型输入。第三项为e2e_time。
  ```shell
  output/$DATE/modeltest/$MODEL_NAME/precision_result/
  ```

## ModelTest性能测试

- 配置性能测试环境变量:
```shell
  export ATB_LLM_BENCHMARK_ENABLE=1
  export ATB_LLM_BENCHMARK_FILEPATH=${script_path}/benchmark.csv
```

- 性能测试时需要在 `${image_path}` 下存放一张图片，使用以下命令运行 `run_pa.sh`。

```shell
bash ${script_path}/run_pa.sh --performance --trust_remote_code ${weight_path} ${image_path} ${batch_size} ${max_input_length} ${max_output_length}
```
注：性能测试结果保存在 ${script_path} 目录下


## Benchmark精度测试
- 首先按照[服务化推理](#服务化推理)，启动服务端进程

- 新建窗口，配置MindIE环境变量，运行如下benchmark命令
```shell
# 输出Benchmark运行日志
export MINDIE_LOG_TO_STDOUT="benchmark:1; client:1" 

benchmark \
--TestAccuracy True \
--DatasetPath ${textvqa_val.jsonl的绝对路径} \
--DatasetType textvqa  \
--ModelName ${与服务化config.json中的modelName保持一致} \
--ModelPath ${模型权重的绝对路径} \
--TestType client \
--Concurrency 64 \
--RequestRate 64 \
--TaskKind stream \
--Tokenizer True \
--MaxOutputLen 20  \
--WarmupSize 1 \
--DoSampling False \
--TrustRemoteCode True \
--Http http://127.0.0.1:${端口号，与起服务化时的config.json保持一致} \
--ManagementHttp http://127.0.0.2:${管理端口号，与起服务化时的config.json保持一致} \
--SavePath ${日志输出路径}
```

## Benchmark性能测试
- 首先按照[服务化推理](#服务化推理)，启动服务端进程

- 新建窗口，配置MindIE环境变量，运行如下benchmark命令
```shell
# 输出benchmark运行日志
export MINDIE_LOG_TO_STDOUT="benchmark:1; client:1" 

benchmark \
--TestAccuracy False \
--DatasetPath ${performance.jsonl的绝对路径} \
--DatasetType textvqa  \
--ModelName ${与服务化config.json中的modelName保持一致} \
--ModelPath ${模型权重的绝对路径} \
--TestType client \
--Concurrency 64 \ # Client模式下，不超过endpoint所支持的最大连接数。取值范围：[1，1000]，默认值：128。
--RequestRate 64 \ # 指定一组发送频率，按照Distribution参数设置的模式进行发送，以每个频率完成一次推理，频率的取值范围：（0，10000]。
--TaskKind stream \
--Tokenizer True \
--MaxOutputLen 256  \
--WarmupSize 1 \
--DoSampling False \
--TrustRemoteCode True \
--Http http://127.0.0.1:${端口号，与起服务化时的config.json保持一致} \
--ManagementHttp http://127.0.0.2:${端口号，与起服务化时的config.json保持一致} \
--SavePath ${日志输出路径}
```