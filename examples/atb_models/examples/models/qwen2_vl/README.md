# README

- Qwen系列多模态模型隶属于阿里巴巴开源的多模态大模型家族，包括Qwen2-VL、Qwen2.5-VL和QvQ等模型，支持文本、图像、音频和视频的输入与处理，具备强大的视觉理解和跨模态交互能力。

## 特性矩阵

| 模型及参数量       | 800I A2 64G Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | MindIE Service | 纯模型支持模态  | 服务化支持模态  |
|--------------|----------------------------|-----------------------------|------|--------------|----------------|----------|----------|
| Qwen2-VL-2B-Instruct  | 支持world size 1,2,4,8         |  不支持                       | √    | √            | √              | 文本、图片、视频 | 文本、图片、视频 |
| Qwen2-VL-2B  | 支持world size 1,2,4,8         | 不支持                        | √    | √            | √              | 文本、图片、视频 | 文本、图片、视频 |
| Qwen2-VL-7B-Instruct  | 支持world size 1,2,4,8         | 不支持                        | √    | √            | √              | 文本、图片、视频 | 文本、图片、视频 |
| Qwen2-VL-7B  | 支持world size 1,2,4,8         | 不支持                        | √    | √            | √              | 文本、图片、视频 | 文本、图片、视频 |
| Qwen2-VL-72B-Instruct | 支持world size 4,8           | 不支持                        | √    | √            | √              | 文本、图片、视频 | 文本、图片、视频 |
| Qwen2-VL-72B | 支持world size 4,8           | 不支持                        | √    | √            | √              | 文本、图片、视频 | 文本、图片、视频 |
| QVQ-72B-Preview | 支持world size 4,8           | 不支持                        | √    | √            | √              | 文本、图片、视频 | 文本、图片、视频 |
| Qwen2.5-VL-3B-Instruct  | 支持world size 1,2,4,8         | 支持world size 1,2,4,8                        | √    | √            | √              | 文本、图片、视频 | 文本、图片、视频 |
| Qwen2.5-VL-7B-Instruct  | 支持world size 1,2,4,8         | 支持world size 1,2,4,8                        | √    | √            | √              | 文本、图片、视频 | 文本、图片、视频 |
| Qwen2.5-VL-32B-Instruct  | 支持world size 4,8         | 支持world size 4,8                        | √    | √            | √              | 文本、图片、视频 | 文本、图片、视频 |
| Qwen2.5-VL-72B-Instruct | 支持world size 4,8           | 支持world size 8                       | √    | √            | √              | 文本、图片、视频 | 文本、图片、视频 |

注意：

- 运行QVQ-72B-Preview以及Qwen2.5-VL系列模型时只需要替换模型权重路径，推理脚本以及精度测试脚本与运行Qwen2-VL模型的文件保持一致。
- 表中所示支持的world size为建议配置，实际运行时还需考虑单卡的显存上限，以及输入序列长度。
- 推理默认加载BF16权重，若要使用FP16格式进行推理，需将权重路径下config.json文件的 `torch_dtype` 属性的值修改为 `float16` 。
- Qwen2.5-VL系列模型使用300I DUO推理时，仅支持FP16权重。

## 路径变量解释

| 变量名        | 含义                                                                           |
|------------|------------------------------------------------------------------------------|
| llm_path   | 模型仓路径，若使用模型仓安装包，则该路径为安装包解压后的路径；若使用源码编译，则路径为 `MindIE-LLM/examples/atb_models`; 若使用镜像运行，则路径为 `/usr/local/Ascend/atb-models`。|

## 快速上手
### 权重下载

- [Qwen2-VL-2B-Instruct](https://modelscope.cn/models/Qwen/Qwen2-VL-2B-Instruct/files)
- [Qwen2-VL-2B](https://modelscope.cn/models/Qwen/Qwen2-VL-2B/files)
- [Qwen2-VL-7B-Instruct](https://modelscope.cn/models/Qwen/Qwen2-VL-7B-Instruct/files)
- [Qwen2-VL-7B](https://modelscope.cn/models/Qwen/Qwen2-VL-7B/files)
- [Qwen2-VL-72B-Instruct](https://modelscope.cn/models/Qwen/Qwen2-VL-72B-Instruct/files)
- [Qwen2-VL-72B](https://modelscope.cn/models/Qwen/Qwen2-VL-72B/files)
- [QVQ-72B-Preview](https://modelscope.cn/models/Qwen/QVQ-72B-Preview/files)
- [Qwen2.5-VL-3B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct/files)
- [Qwen2.5-VL-7B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct/files)
- [Qwen2.5-VL-32B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-VL-32B-Instruct/files)
- [Qwen2.5-VL-72B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-VL-72B-Instruct/files)

### 安装依赖

- Toolkit, MindIE/ATB, ATB-SPEED等，参考[此README文件](../../../README.md)，镜像包中一般已默认安装。
- 安装Python其他第三方库依赖，参考[requirements_qwen2_vl.txt](../../../requirements/models/requirements_qwen2_vl.txt)，注意 `transformers == 4.49.0` 
  ```shell
  pip install -r ${llm_path}/requirements/models/requirements_qwen2_vl.txt
  ```
### 调试建议

- **遇到未知错误时的日志调试建议**  
  若在运行过程中遇到未知报错或异常现象，建议开启日志打印开关以辅助排查问题。
  在终端中执行以下命令设置环境变量：

  ```shell
  export MINDIE_LOG_TO_STDOUT=1
  export MINDIE_LOG_TO_FILE=1
  export MINDIE_LOG_LEVEL=info
  ```

  启用该环境变量后，系统日志将输出至终端（标准输出），并同步保存在'/root/mindie'目录下，方便后续定位和分析问题

### 纯模型推理
- 必要入参说明
    - `--model_path` : 本地权重路径，必须输入
    - `--input_image` : 图片或者视频的本地文件路径
    - `--dataset_path` : 图片或者视频的本地文件夹路径
    - `--max_batch_size` : Batch Size
    - `--max_input_length` : 最大输入长度（需考虑输入图片的分辨率以及视频的长度）
    - `--max_output_length` : 最大输出长度
    - `--input_text` : 输入prompt
    - `--shm_name_save_path` : 共享内存的保存路径，设置为能访问到的txt文件即可
    - 其他支持的推理参数请参考 `${llm_path}/examples/models/qwen2_vl/run_pa.py` 文件
- 如需修改TP卡数，修改 `${llm_path}/examples/models/qwen2_vl/run_pa.sh` 文件中环境变 `ASCEND_RT_VISIBLE_DEVICES` 为指定卡号
    ```shell
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
    ```
- 打开模型日志
  ```shell
  export MINDIE_LOG_TO_STDOUT=1
  export MINDIE_LOG_TO_FILE=1
  export MINDIE_LOG_LEVEL=info
  ```
- 执行启动脚本 `${llm_path}/examples/models/qwen2_vl/run_pa.sh`
    ```shell
    bash ${llm_path}/examples/models/qwen2_vl/run_pa.sh --model_path MODEL_PATH --input_image INPUT_IMAGE
    ```
    或
    ```shell
    bash ${llm_path}/examples/models/qwen2_vl/run_pa.sh --model_path MODEL_PATH --dataset_path DATASET_PATH
    ```

- 环境变量说明 (以下为run_pa.sh启动脚本中配置的环境变量)
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
    - 指定当前机器上可用的逻辑NPU核心，多个核心间使用逗号相连
    - 核心ID查阅方式见[此README文件](../../README.md)的【启动脚本相关环境变量】章节
    - 各模型支持的核心数参考“特性矩阵”
  - `export MASTER_PORT=20036`
    - 设置卡间通信端口
    - 默认使用20036端口
    - 目的是为了避免同一台机器同时运行多个多卡模型时出现通信冲突
    - 设置时端口建议范围为：20000-20050
  - 以下环境变量与性能和内存优化相关，通常情况下无需修改
    ```shell
    export INF_NAN_MODE_ENABLE=0
    export ATB_OPERATION_EXECUTE_ASYNC=1
    export TASK_QUEUE_ENABLE=1
    export ATB_CONVERT_NCHW_TO_ND=1
    ```
### 服务化推理
- 以容器化部署 `MindIE` 为例，打开 `MindIE-Service` 配置文件 `config.json`

  ```shell
  vim /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
  ```

- 修改`MindIE-Service`配置文件`config.json`，以`Qwen2-VL-7B-Instruct`，`Qwen2-VL-72B-Instruct`为例，在800I A2环境下，推荐使用以下配置，请自行修改`modelWeightPath`为实际权重路径：

  - **Qwen2-VL-7B-Instruct**
  ```json
  {
      "ServerConfig": {
          "port" : 1025,
          "managementPort" : 1026,
          "httpsEnabled": false,
      },
      "BackendConfig": {
          "npuDeviceIds": [
              [0,1]
          ],
          "ModelDeployConfig": {
              "maxSeqLen": 32768,
              "maxInputTokenLen": 32768,
              "ModelConfig": [
                  {
                      "modelInstanceType": "Standard",
                      "modelName": "qwen2_vl",
                      "modelWeightPath": "/data/Qwen2-VL-7B-Instruct",
                      "worldSize": 2,
                      "npuMemSize": 10, #KV Cache的显存大小，单位是GB，请勿设置为-1，需要给vit预留空间。
                  }
              ]
          },
          "ScheduleConfig": {
              "maxPrefillTokens": 32768,
          }
      }
  }
  ```

  - **Qwen2-VL-72B-Instruct**
  ```json
  {
      "ServerConfig": {
          "port" : 1025,
          "managementPort" : 1026,
          "httpsEnabled": false,
      },
      "BackendConfig": {
          "npuDeviceIds": [
              [0,1,2,3,4,5,6,7]
          ],
          "ModelDeployConfig": {
              "maxSeqLen": 32768,
              "maxInputTokenLen": 32768,
              "ModelConfig": [
                  {
                      "modelInstanceType": "Standard",
                      "modelName": "qwen2_vl",
                      "modelWeightPath": "/data/Qwen2-VL-72B-Instruct",
                      "worldSize": 8,
                      "npuMemSize": 10, #KV Cache的显存大小，单位是GB，请勿设置为-1，需要给vit预留空间。
                  }
              ]
          },
          "ScheduleConfig": {
              "maxPrefillTokens": 65536,
          }
      }
  }
  ```
  - **QvQ**
  ```json
  {
      "ServerConfig": {
          "port" : 1025,
          "managementPort" : 1026,
          "httpsEnabled": false,
      },
      "BackendConfig": {
          "npuDeviceIds": [
              [0,1,2,3,4,5,6,7]
          ],
          "ModelDeployConfig": {
              "maxSeqLen": 32768,
              "maxInputTokenLen": 32768,
              "ModelConfig": [
                  {
                      "modelInstanceType": "Standard",
                      "modelName": "qvq", #若部署QVQ-72B-Preview模型，modelName建议设置为qvq，测试工具会根据此模型名称进行答案过滤，方便计算数据集精度
                      "modelWeightPath": "/data/QVQ-72B-Preview",
                      "worldSize": 8,
                      "npuMemSize": 10, #KV Cache的显存大小，单位是GB，请勿设置为-1，需要给vit预留空间。
                  }
              ]
          },
          "ScheduleConfig": {
              "maxPrefillTokens": 65536,
              "maxIterTimes": 2048, # 模型最大输出长度
          }
      }
  }
  ```

- 部署服务化
  - 在部署服务化的终端设置模型日志开启环境变量
    ```shell
    export MINDIE_LOG_TO_STDOUT=1
    export MINDIE_LOG_TO_FILE=1
    export MINDIE_LOG_LEVEL=info
    ```
  - 部署
    ```shell
    cd /usr/local/Ascend/mindie/latest/mindie-service/bin
    ./mindieservice_daemon
    ```

- 新建一个Docker终端会话，发送curl请求完成推理，以OpenAI接口与vLLM接口为例

  - **OpenAI接口**
  ```shell
  curl http://localhost:${端口号，与起服务化时config.json中的'port'保持一致}/v1/chat/completions -d '{
    "model": "qwen2_vl",
    "messages": [{
      "role": "user",
      "content": [
                  {
                      "type": "text",
                      "text": "Explain the contents of the picture."
                  },
                  {"type": "image_url", "image_url": ${图片路径}}                        
              ]
    }],
    "max_tokens": 512,
    "stream": false
  }'
  ```

  - **vLLM接口**
  ```shell
  curl localhost:${端口号，与起服务化时config.json中的'port'保持一致}/generate -d '{
      "prompt": [
          {"type": "text", "text": "Explain the contents of the picture."},
          {
              "type": "image_url",
              "image_url": ${图片路径}
          }
      ],
      "max_tokens": 512,
      "do_sample": false,
      "stream": false,
      "model": "qwen2_vl"
  }'
  ```

## 精度测试（纯模型推理场景）
### TextVQA

- 数据准备
    - 数据集下载 [textvqa](https://huggingface.co/datasets/maoxx241/textvqa_subset)
    - 保证 `textvqa_val.jsonl` 和 `textvqa_val_annotations.json` 在同一目录下
    - 将 `textvqa_val.jsonl` 文件中所有 `image` 属性的值改为相应图片的绝对路径
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
- 安装`modeltest`及其三方依赖
  ```shell
  pip install --upgrade pip
  pip install -e .
  pip install tabulate termcolor 
  ```
- 将 `modeltest/config/model/qwen2_vl.yaml` 中的 `model_path` 的值修改为模型权重的绝对路径
  ```yaml
  model_path: /data_mm/weights/Qwen2-VL-7B-Instruct
  ```
- 将 `modeltest/config/task/textvqa.yaml` 中的 `local_dataset_path` 修改为textvqa_val.jsonl文件的绝对路径
  ```yaml
  local_dataset_path: /data_mm/datasets/textvqa_val/textvqa_val.jsonl
  ```
- 设置可见卡数，修改 `scripts/mm_run.sh` 文件中的 `ASCEND_RT_VISIBLE_DEVICES` 。依需求设置单卡或多卡可见。
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1
  ```
- 运行测试命令
  ```shell
  bash scripts/mm_run.sh textvqa qwen2_vl
  ```
- 测试结果保存于以下路径。其下的 `results/..(一系列文件夹嵌套)/\*\_result.csv` 中存放着modeltest的测试结果；`debug/..(一系列文件夹嵌套)/output\_\*.txt` 中存储着每一条数据的运行结果。
  ```shell
  output/$DATE/modeltest/$MODEL_NAME/precision_result/
  ```
### VideoBench

- 数据准备
  - 数据集下载 [Eval_QA](https://huggingface.co/datasets/maoxx241/videobench_subset) && [Video-Bench](https://huggingface.co/datasets/LanguageBind/Video-Bench/tree/main)
  - 将`Eval_QA/`目录下的各json文件中的`vid_path`属性的值改为相应图片的绝对路径
  ```json
  ...
  "v_C7yd6yEkxXE_4": {
    "vid_path": "/data_mm/Eval_video/ActivityNet/v_C7yd6yEkxXE.mp4",
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
- 安装`modeltest`及其三方依赖
  ```shell
  pip install --upgrade pip
  pip install -e .
  pip install tabulate termcolor 
  ```
- 将 `modeltest/config/model/qwen2_vl.yaml` 中的 `model_path` 的值修改为模型权重的绝对路径
  ```yaml
  model_path: /data_mm/weights/Qwen2-VL-7B-Instruct
  ```
- 将 `modeltest/config/task/videobench.yaml` 中的 `local_dataset_path` 修改为 `Video-Bench-main/Eval_QA` 文件夹的绝对路径，调整输入长度`requested_max_input_length`，查看 `EVAL_QA` 文件夹下的json文件，将 `subject_mapping` 中不涉及测试的视频子数据集注释掉（可自行调整），样例如下：
  ```yaml
  local_dataset_path: /data_mm/datasets/VideoBench/Video-Bench-main/Eval_QA
  ...
  requested_max_input_length: 30000
  ...
  subject_mapping:
  # ActivityNet:
  #  name: ActivityNet
  Driving-decision-making:
    name: Driving-decision-making
  ```
- 设置可见卡数，修改 `scripts/mm_run.sh` 文件中的 `ASCEND_RT_VISIBLE_DEVICES` 。依需求设置单卡或多卡可见。
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1
  ```
- 运行测试命令
  ```shell
  bash scripts/mm_run.sh videobench qwen2_vl
  ```
- 测试结果保存于以下路径。其下的 `results/..(一系列文件夹嵌套)/\*\_result.csv` 中存放着modeltest的测试结果，`debug/..(一系列文件夹嵌套)/output\_\*.txt` 中存储着每一条数据的运行结果。
  ```shell
  output/$DATE/modeltest/$MODEL_NAME/precision_result/
  ```

## 精度测试（服务化推理场景）

### TextVQA

- 按照纯模型推理场景中TextVQA部分指导准备数据集

- 参考[服务化推理](#服务化推理)章节，在当前Docker容器中部署并启动推理服务。

- 新建一个Docker终端会话，使用benchmark工具进行数据集测试，执行如下命令：

  - 打开benchmark工具日志日志打印开关
    ```shell
    export MINDIE_LOG_TO_STDOUT="benchmark:1; client:1" 
    ```

  - 发送benchmark推理请求
    ```shell
    benchmark \
    --TestAccuracy True \
    --DatasetPath ${数据集中textvqa_val.jsonl的绝对路径} \
    --DatasetType textvqa  \
    --ModelName qwen2_vl \
    --ModelPath ${模型权重的绝对路径} \
    --TestType client \
    --Concurrency 1 \
    --RequestRate 1 \
    --TaskKind stream \
    --Tokenizer True \
    --MaxOutputLen 20  \
    --WarmupSize 1 \
    --DoSampling False \
    --Http http://127.0.0.1:${端口号，与起服务化时config.json中的'port'保持一致} \
    --ManagementHttp http://127.0.0.2:${端口号，与起服务化时的config.json中的'managementPort'保持一致} \
    --SavePath ${日志输出路径}
    ```

- 完成数据集推理后，测试结果将打印展示数据集得分等指标，同时测试结果会保存在 `--SavePath` 路径下。

### VideoBench

- 按照纯模型推理场景中VideoBench部分指导准备数据集

- 参考[服务化推理](#服务化推理)章节，在当前Docker容器中部署并启动推理服务。

- 新建一个Docker终端会话，使用benchmark工具进行数据集测试，执行如下命令：

  - 打开benchmark工具日志打印开关
    ```shell
    export MINDIE_LOG_TO_STDOUT="benchmark:1; client:1" 
    ```

  - 发送benchmark推理请求
    ```shell
    benchmark \
    --TestAccuracy True \
    --DatasetPath ${数据集中Eval_QA文件夹的绝对路径} \
    --DatasetType videobench  \
    --ModelName qwen2_vl \
    --ModelPath ${模型权重的绝对路径} \
    --TestType client \
    --Concurrency 1 \
    --RequestRate 1 \
    --TaskKind stream \
    --Tokenizer True \
    --MaxOutputLen 20  \
    --WarmupSize 1 \
    --DoSampling False \
    --Http http://127.0.0.1:${端口号，与起服务化时config.json中的'port'保持一致} \
    --ManagementHttp http://127.0.0.2:${端口号，与起服务化时的config.json中的'managementPort'保持一致} \
    --SavePath ${日志输出路径}
    ```

- 完成数据集推理后，测试结果将打印展示数据集得分等指标，同时测试结果会保存在 `--SavePath` 路径下。

## 性能测试（纯模型推理场景）

- 打开模型日志
  ```shell
  export MINDIE_LOG_TO_STDOUT=1
  export MINDIE_LOG_TO_FILE=1
  export MINDIE_LOG_LEVEL=info
  ```
- 使用 `${llm_path}/examples/models/qwen2_vl/run_pa.sh` 进行纯模型推理测试。
- 设置 `--max_output_length` 为合理值（默认256），确保实际输出文本长度 ≥ 该值。
- 使用 `--input_image` + `--max_batch_size`：测试指定batch数量的单张图片性能。
- 使用 `--dataset_path` + `--max_batch_size`：批量处理指定文件夹内所有图片。
- 纯模型测试使用固定prompt，如需测试自定义数据集，请使用服务化推理场景进行性能测试，详见下文。
- 参考[纯模型推理](#纯模型推理)章节，执行`${llm_path}/examples/models/qwen2_vl/run_pa.sh`推理脚本，查看终端输出的性能数据。

## 性能测试（服务化推理场景）

- 下载[textvqa_val_performance](https://huggingface.co/datasets/maoxx241/textvqa_val_performance)数据集，如要测试自定义数据集，自行修改下载好的 `textvqa_val_performance` 数据集中 `textvqa_val.jsonl` 里面的 `"image"` 属性的值（图片路径），`"question"` 属性的值（问题）。
- 参考[服务化推理](#服务化推理)章节，在当前Docker容器中部署并启动推理服务。

- 新建一个Docker终端会话，使用benchmark工具进行数据集测试，执行如下命令：

  - 打开benchmark工具日志打印开关
    ```shell
    export MINDIE_LOG_TO_STDOUT="benchmark:1; client:1" 
    ```

  - 发送benchmark推理请求
    ```shell
    benchmark \
    --TestAccuracy False \
    --DatasetPath ${textvqa_val_performance数据集中textvqa_val.jsonl的绝对路径} \
    --DatasetType textvqa  \
    --ModelName qwen2_vl \
    --ModelPath ${模型权重的绝对路径} \
    --TestType client \
    --Concurrency 64 \
    --RequestRate 64 \
    --TaskKind stream \
    --Tokenizer True \
    --MaxOutputLen 256  \
    --WarmupSize 1 \
    --DoSampling False \
    --Http http://127.0.0.1:${端口号，与起服务化时config.json中的'port'保持一致} \
    --ManagementHttp http://127.0.0.2:${端口号，与起服务化时的config.json中的'managementPort'保持一致} \
    --SavePath ${日志输出路径}
    ```

- 完成数据集推理后，测试结果将打印展示吞吐(GenerationSpeed)等性能指标，同时测试结果会保存在--SavePath路径下。

## 限制与约束

- 当用户使用多模态理解模型在服务化推理，且输入输出序列为长序列时，可能会由于超出NPU内存限制而推理失败。以 800I A2 64G 8卡，输入tokens=16384，输出tokens=2048场景为例，需要使用以下服务化参数进行配置。当用户使用的序列更长时，可以适当下调并发数。

| 模型名称 | maxPrefillTokens | npuMemSize | 最大并发数  |
|--------------|--------------|--------------|--------------|
| Qwen2-VL-72B-Instruct | 16384 | 30 | 40 |
| Qwen2.5-VL-72B-Instruct | 16384 | 30 | 40 |