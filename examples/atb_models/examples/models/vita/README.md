# README

- [VITA](https://github.com/VITA-MLLM/VITA)是第一个开源的多模态大模型（MLLM），擅长同时处理和分析视频、图像、文本和音频模态，同时具有先进的多模态交互体验。
- 此代码仓中实现了一套基于NPU硬件的VITA推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

## 特性矩阵
| 模型及参数量      | 800I A2 64GB | 300I DUO | FP16 | BF16 | MindIE Service |纯模型支持模态  | 服务化支持模态 |
|-------------|----------------------------|-----------------------------|------|------------------|-----------------|-----|-----|
|  vita-1.5(Qwen2)    | 支持world size 1     | 不支持        | x   |  √                   | √              | 文本、图片、音频、视频           | 文本、图片、音频、视频|

须知：
1. 当前版本服务化仅支持单个请求单张图片输入
2. 当前多模态场景，MindIE Service仅支持MindIE Service、TGI、Triton、vLLM Generate、OpenAI 5种服务化请求格式

## 路径变量解释

| 变量名      | 含义                                                                                                                                                         |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| working_dir | 加速库及模型库下载后放置的目录                                                                                                                               |
| llm_path    | 模型仓所在路径。若使用编译好的包，则路径为 `${working_dir}/MindIE-LLM/`；若使用gitcode下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径；vita的工作脚本所在路径为 `${llm_path}/examples/models/vita`                                                                                           |
| weight_path | 模型权重路径
| audio_path  | 音频所在路径                                                                      |
| image_path  | 图片所在路径                                                                      |
| video_path  | 视频所在路径                                                                      |
| max_batch_size  | 最大bacth数                                                                  |
| max_input_length  | 多模态模型的最大embedding长度，                                             |
| max_output_length | 生成的最大token数                                                          |

-注意：
max_input_length长度设置可参考模型权重路径下config.json里的max_position_embeddings参数值
## 权重

**权重下载**

- [VITA](https://huggingface.co/VITA-MLLM/VITA-1.5/tree/main)
- [InternViT-300M-448px](https://huggingface.co/OpenGVLab/InternViT-300M-448px/tree/main)
-权重准备：
  在weight_path执行：
  ```shell
  cd VITA
  mv ../InternViT-300M-448px ./VITA_ckpt
  ```
  并将VITA权重中config.json的mm_vision_tower字段的参数修改为"InternViT-300M-448px"
- 在大batch size场景下，需要将VITA权重中config,json的参数tokenizer_model_max_length修改为32768(对应max_position_embeddings)

**基础环境变量**

1、安装 CANN 8.0 的环境，并 `source /path/to/cann/set_env.sh`；

2、使用 Python 3.9 或更高；

3、使用 torch 2.0 或更高版本，并安装对应的 torch_npu；

4、安装依赖：

- Python其他第三方库依赖，参考[requirements_vita.txt](../../../requirements/models/requirements_vita.txt)
- 参考[此README文件](../../../README.md)

## 推理

### 对话测试

**运行Paged Attention FP16**

- 运行启动脚本
  - 在\${llm_path}目录下执行以下指令
    ```shell
    bash ${script_path}/run_pa.sh --run ${weight_path} --image_path ${image_path} --audio_path ${audio_path}
    bash ${script_path}/run_pa.sh --run ${weight_path} --video_path ${video_path} --question "你是谁？"
    ```
  - 注意：
    音频和文本不能同时设置
- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0`
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

## 推理服务化部署
  如果要将该模型用于高性能推理服务部署，实现大模型推理性能测试、精度测试和可视化能力，可参考下述参数配置和启动服务方式
- 环境准备
  修改MindIE-Service配置文件进行参数配置：

  ```shell
  vim /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
  ```

  在800I A2环境下，推荐使用以下配置，请自行修改"modelWeightPath"为实际权重路径：

  ```json
  {
      "ServerConfig": {
          "port" : 1025,
          "managementPort" : 1026,
          "httpsEnabled": false,
      },
      "BackendConfig": {
          "npuDeviceIds": [
              [0]
          ],
          "ModelDeployConfig": {
              "maxSeqLen": 32768,
              "maxInputTokenLen": 32768,
              "ModelConfig": [
                  {
                      "modelInstanceType": "Standard",
                      "modelName": "vita",
                      "modelWeightPath": "/data/datasets/VITA",
                      "worldSize": 1,
                      "npuMemSize": -1, #kvcache分配，可自行调整，单位是GB
                      "trustRemoteCode": false #默认为false，若设为true，则信任本地代码，用户需自行承担风险
                  }
              ]
          },
          "ScheduleConfig": {
              "maxPrefillTokens": 32768,
          }
      }
  }
  ```

- 启动服务
  可在命令行直接启动：
  ```shell
  cd /usr/local/Ascend/mindie/latest/mindie-service/bin
  ./mindieservice_daemon
  ```
  回显如下则说明启动成功。
  ```shell
  Daemon start success!
  ```

- 另起一个新的会话并进入同一环境中（docker），发送curl请求完成推理，注意：在发送请求的prompt里，必须image或video在前，audio或text在后，以OpenAI接口与vLLM接口为例

  **OpenAI接口**
  ```shell
  curl http://localhost:${端口号，与起服务化时config.json中的'port'保持一致}/v1/chat/completions -d '{
    "model": "vita",
    "messages": [{
      "role": "user",
      "content": [
                  {"type": "image_url", "image_url": ${图片路径}},
                  {
                      "type": "text",
                      "text": "Explain the contents of the picture."
                  }                 
              ]
    }],
    "max_tokens": 512,
    "stream": false
  }'
  ```

  **vLLM接口**
  ```shell
  curl localhost:${端口号，与起服务化时config.json中的'port'保持一致}/generate -d '{
      "prompt": [
          {
              "type": "image_url",
              "image_url": ${图片路径}
          }，
          {"type": "audio_url", "audio_url": ${音频路径}}
      ],
      "max_tokens": 512,
      "do_sample": false,
      "stream": false,
      "model": "vita"
  }'
  ```
## 精度与性能测试方案
为了评估模型在处理不同类型数据（如文本、图像、视频等）时的效果与表现，我们为MindIE中的服务化推理场景分别准备了对应的精度与性能测试方案供用户参考，以下是方案的具体实现。

### 纯模型推理场景 性能测试

- 开启环境变量，并将 ${script_path}/vita.py 中的 PERF_FILE 修改为结果保存的路径
  ```shell
  export ATB_LLM_BENCHMARK_ENABLE=1
  ```
- 运行启动脚本，会自动输出batchsize为1-10时的吞吐，并将结果保存在上述修改后的路径中
  ```shell
  bash ${script_path}/run_pa.sh --performance ${weight_path} --image_path ${image_path} --audio_path ${audio_path}
  bash ${script_path}/run_pa.sh --performance ${weight_path} --video_path ${video_path} --question "你是谁？"
  ```

### 服务化推理场景 精度测试
首先按照[推理服务化部署](#推理服务化部署)，启动服务

#### TextVQA 图片+文本理解场景

- 数据准备
    - 数据集下载 [textvqa_val](https://huggingface.co/datasets/maoxx241/textvqa_val)
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

另起一个新的会话并进入同一环境中（docker），使用benchmark工具进行数据集测试，执行如下benchmark命令：

- 打开benchmark工具打印开关
  ```shell
  export MINDIE_LOG_TO_STDOUT="benchmark:1; client:1" 
  ```

- 发送benchmark推理请求（若出现trust_remote_code相关报错，需将--TrustRemoteCode置为True，可能引入文件读取风险，请知悉）
  ```shell
  benchmark \
  --TestAccuracy True \
  --DatasetPath ${数据集中textvqa_val.jsonl的绝对路径} \
  --DatasetType textvqa  \
  --ModelName vita \
  --ModelPath ${模型权重的绝对路径} \
  --TestType client \
  --Concurrency 1 \
  --RequestRate 1 \
  --TaskKind stream \
  --Tokenizer True \
  --MaxOutputLen 20  \
  --WarmupSize 1 \
  --DoSampling False \
  --TrustRemoteCode False \
  --Http http://127.0.0.1:${端口号，与参数配置时config.json中的'port'保持一致} \
  --ManagementHttp http://127.0.0.2:${端口号，与参数配置时的config.json中的'managementPort'保持一致} \
  --SavePath ${日志输出路径}
  ```

完成数据集推理后，测试结果将打印展示数据集得分等指标，同时测试结果会保存在--SavePath路径下。

#### VideoBench视频+文本理解场景

- 数据准备
  - 数据集下载 [Eval_QA](https://huggingface.co/datasets/maoxx241/videobench_subset) && [Video-Bench](https://huggingface.co/datasets/LanguageBind/Video-Bench/tree/main)
  - 将Eval_QA/目录下的各json文件中的vid_path改为相应视频的绝对路径
  ```json
  ...
  "v_C7yd6yEkxXE_4": {
    "vid_path": "/data_mm/Eval_video/ActivityNet/v_C7yd6yEkxXE.mp4",
  }
  ...
  ```

另起一个新的session并进入同一环境中（docker），使用benchmark工具进行数据集测试，执行如下benchmark命令：

- 打开benchmark工具打印开关
  ```shell
  export MINDIE_LOG_TO_STDOUT="benchmark:1; client:1" 
  ```

- 发送benchmark推理请求（若出现trust_remote_code相关报错，需将--TrustRemoteCode置为True，可能引入文件读取风险，请知悉）
  ```shell
  benchmark \
  --TestAccuracy True \
  --DatasetPath ${数据集中Eval_QA文件夹的绝对路径} \
  --DatasetType videobench  \
  --ModelName vita \
  --ModelPath ${模型权重的绝对路径} \
  --TestType client \
  --Concurrency 1 \
  --RequestRate 1 \
  --TaskKind stream \
  --Tokenizer True \
  --MaxOutputLen 20  \
  --WarmupSize 1 \
  --DoSampling False \
  --TrustRemoteCode False \
  --Http http://127.0.0.1:${端口号，与参数配置时config.json中的'port'保持一致} \
  --ManagementHttp http://127.0.0.2:${端口号，与参数配置时的config.json中的'managementPort'保持一致} \
  --SavePath ${日志输出路径}
  ```

完成数据集推理后，测试结果将打印展示数据集得分等指标，同时测试结果会保存在--SavePath路径下。

### 服务化推理场景 性能测试
首先按照[推理服务化部署](#推理服务化部署)，启动服务，并完成[textvqa_val_performance](https://huggingface.co/datasets/maoxx241/textvqa_val_performance)数据集下载

另起一个新的session并进入同一环境中（docker），使用benchmark工具进行数据集测试，执行如下benchmark命令：

- 打开benchmark工具打印开关
  ```shell
  export MINDIE_LOG_TO_STDOUT="benchmark:1; client:1" 
  ```

- 发送benchmark推理请求（若出现trust_remote_code相关报错，需将--TrustRemoteCode置为True，可能引入文件读取风险，请知悉）
  ```shell
  benchmark \
  --TestAccuracy False \
  --DatasetPath ${textvqa_val_performance数据集中textvqa_val.jsonl的绝对路径} \
  --DatasetType textvqa  \
  --ModelName vita \
  --ModelPath ${模型权重的绝对路径} \
  --TestType client \
  --Concurrency 64 \
  --RequestRate 64 \
  --TaskKind stream \
  --Tokenizer True \
  --MaxOutputLen 256  \
  --WarmupSize 1 \
  --DoSampling False \
  --TrustRemoteCode False \
  --Http http://127.0.0.1:${端口号，与参数配置时config.json中的'port'保持一致} \
  --ManagementHttp http://127.0.0.2:${端口号，与参数配置时的config.json中的'managementPort'保持一致} \
  --SavePath ${日志输出路径}
  ```

完成数据集推理后，测试结果将打印展示吞吐(GenerationSpeed)等性能指标，同时测试结果会保存在--SavePath路径下。

## FAQ
- 在对话测试或者精度测试时，用户如果需要修改输入input_texts,max_batch_size时，可以修改`${script_path}/vita.py`里的参数，具体可见vita.py
- 更多环境变量见[此README文件](../../README.md)
- 服务化集成部署更多信息可参考昇腾社区 MindIE 服务化集成部署章节

