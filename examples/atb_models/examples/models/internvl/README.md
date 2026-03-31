# README

- [InternVL](https://github.com/OpenGVLab/InternVL)，是一种多模态大模型，具有强大的图像和文本处理能力，通过开源组件缩小与商业多模态模型的差距——GPT-4V的开源替代方案。在聊天机器人中，InternVL可以通过解析用户的文字输入，结合图像信息，生成更加生动、准确的回复。 此外，InternVL还可以根据用户的图像输入，提供相关的文本信息，实现更加智能化的交互。
- 此代码仓中实现了一套基于NPU硬件的InternVL推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。
- 支持 InternVL 系列中的大部分开源模型，具体支持情况请参见[特性矩阵](#特性矩阵)部分。

## 特性矩阵

- 此矩阵罗列了各InternVL模型支持的特性

| 模型及参数量  | 800I A2 64GB | 300I DUO | FP16 | BF16 | MindIE Service | 纯模型支持模态 | 服务化支持模态 |
| ------------- |--------------------------|---------------------------| ---- | ---- | --------------- | --------------- | -------- |
| InternVL2-8B | 支持world size 1,2,4,8     | 不支持 | √    | ×    | √               | 文本、图片               | 文本、图片        | 
| InternVL2-40B | 支持world size 2,4,8         | 不支持 | √    | √    | √               | 文本、图片               | 文本、图片        |
| InternVL2.5-8B | 支持world size 1,2,4,8         | 不支持 | √    | ×    | √               | 文本、图片               | 文本、图片        |
| InternVL2.5-78B | 支持world size 8         | 不支持 | √    | √    | √               | 文本、图片               | 文本、图片        |

注意：

- 当前多模态场景, MindIE Service仅支持MindIE Service、TGI、Triton、vLLM Generate 4种服务化请求格式。
- 表中所示支持的world size为建议配置，实际运行时还需考虑单卡的显存上限，以及输入序列长度。
- 推理默认加载BF16权重，如运行特性矩阵中不支持BF16的模型，请将权重路径下config.json文件的`torch_dtype`字段修改为`float16`。
- MindIE Service表示模型支持MindIE服务化部署，多卡服务化推理场景。
- 若需要在同一环境下拉起多个 MindIE Service 服务，需要设置以下环境变量，确保每个服务之间的 MASTER_PORT 不冲突。该变量的取值范围通常为 [1024, 65535]，建议动态分配一个未占用的端口。

  ```shell
  export MASTER_ADDR=localhost
  export MASTER_PORT=<动态分配的未占用端口号>
  ```

  提示：可以通过以下命令检查某个端口是否被占用：

  ```shell
  netstat -anp | grep <PORT>
  ```

## 路径变量解释

| 变量名      | 含义                                                                                                                    |
| ----------- |-----------------------------------------------------------------------------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                                                                                                       |
| llm_path    | 模型仓所在路径：`${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径；internvl的工作脚本所在路径为 `${llm_path}/examples/models/internvl`                                                      |
| weight_path | 模型权重路径                                                                                                                |
| trust_remote_code | 是否信任模型权重路径下的可执行文件：默认不信任，若传入此参数则信任，**用户需自行承担风险**                                                                 |
| image_path  | 图片所在路径                                                                                                                |

## 权重

**权重下载**

- [InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B/tree/main)
- [InternVL2-40B](https://huggingface.co/OpenGVLab/InternVL2-40B/tree/main)
- [InternVL2_5-8B](https://huggingface.co/OpenGVLab/InternVL2_5-8B/tree/main)
- [InternVL2_5-78B](https://huggingface.co/OpenGVLab/InternVL2_5-78B/tree/main)

**基础环境变量**

- 1.Python其他第三方库依赖，参考[requirements_internvl.txt](../../../requirements/models/requirements_internvl.txt)
- 2.参考[此README文件](../../../README.md)
- 注意：保证先后顺序，首先安装FrameworkPTAdapter中的pytorch和torch_npu，再安装其他的python依赖。

## 推理

### 安装依赖

- Toolkit, MindIE/ATB, ATB-SPEED等，参考[此README文件](../../../README.md)，镜像包中一般已默认安装
- 安装Python其他第三方库依赖，参考[requirements_internvl.txt](../../../requirements/models/requirements_internvl.txt)

  ```shell
  pip install -r ${llm_path}/requirements/models/requirements_internvl.txt
  ```

- 若下载有SSL相关报错，可在命令后加上 `-i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com` 参数使用阿里源进行下载

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

- 运行启动脚本
  - 在`${llm_path}`目录下执行以下指令

    ```shell
    bash ${script_path}/run_pa.sh --run --trust_remote_code ${weight_path} ${image_path}
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

## 服务化推理

- 打开配置文件

  ```shell
  vim /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
  ```

- 修改`MindIE-Service`配置文件`config.json`，以`InternVL2_5-8B`，`InternVL2_5-78B`为例，在800I A2环境下，推荐使用以下配置，请自行修改`modelWeightPath`为实际权重路径：

  - **InternVL2_5-78B**

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
                      "modelName": "internvl",
                      "modelWeightPath": "/data/datasets/InternVL2_5-78B",
                      "worldSize": 8,
                      "npuMemSize": 20, #kvcache分配，可自行调整，单位是GB
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

  - **InternVL2_5-8B**

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
                      "modelName": "internvl",
                      "modelWeightPath": "/data/datasets/InternVL2_5-8B",
                      "worldSize": 1,
                      "npuMemSize": 24, #kvcache分配，可自行调整，单位是GB
                      "trustRemoteCode": true #默认为false，该模型需要设为true，信任本地代码，若使用该推荐配置，用户需自行承担风险
                  }
              ]
          },
          "ScheduleConfig": {
              "maxPrefillTokens": 65536,
          }
      }
  }
  ```

- 部署服务化

  ```shell
  cd /usr/local/Ascend/mindie/latest/mindie-service/bin
  ./mindieservice_daemon
  ```

- 新建同一个Docker容器的终端会话，在任意路径下发送curl请求完成推理，以下分别以OpenAI接口与vLLM接口为例

  - **OpenAI接口**

  ```shell
  curl http://localhost:${端口号，与起服务化时config.json中的'port'保持一致}/v1/chat/completions -d '{
    "model": "internvl",
    "messages": [{
      "role": "user",
      "content": [
                  {
                      "type": "text",
                      "text": "Explain the contents of the picture."
                  },
                  {"type": "image_url", "image_url": "${图片路径}}"                        
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
              "image_url": "${图片路径}"
          }
      ],
      "max_tokens": 512,
      "do_sample": false,
      "stream": false,
      "model": "internvl"
  }'
  ```

## 精度与性能测试方案

为了全面评估模型在处理不同类型数据（如文本、图像、视频等）时的效果与表现，我们为MindIE中的纯模型推理与服务化推理场景分别准备了对应的精度与性能测试方案供用户参考，以下是方案的具体实现。

## 纯模型推理场景 精度测试

### TextVQA 图片+文本理解场景

- 数据准备
    - 数据集下载 [textvqa_val](https://huggingface.co/datasets/maoxx241/textvqa_val)
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

- 进入以下目录 ${llm_path}/examples/atb_models/tests/modeltest

  ```shell
  cd ${llm_path}/examples/atb_models/tests/modeltest
  ```

- 安装`modeltest`及其三方依赖
 
  ```shell
  pip install --upgrade pip
  pip install -e .
  pip install tabulate termcolor 
  ```

   - 若下载有SSL相关报错，可在命令后加上 `-i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com` 参数使用阿里源进行下载
- 修改modeltest/config/model/internvl.yaml中配置项
    - 将`model_path`的值修改为模型权重的绝对路径
    - 将warm_up_image_path改为textvqa数据集中任一图片的绝对路径
    - 将trust_remote_code修改为 `True` (注意：trust_remote_code为可选参数代表是否信任本地的可执行文件，默认为false。若设置为true，则信任本地可执行文件，此时transformers会执行用户权重路径下的代码文件，这些代码文件的功能的安全性需由用户保证。)

  ```yaml
  model_path: /data_mm/weights/InternVL2-8B 
  trust_remote_code: {用户输入的trust_remote_code值}
  mm_model:
    warm_up_image_path: ['/data_mm/datasets/textvqa_val/train_images/003a8ae2ef43b901.jpg']
  ```

- 修改 `modeltest/config/task/textvqa.yaml` 中配置项
    - 将local_dataset_path的值修改为数据集中textvqa_val.jsonl文件的绝对路径
    - 将requested_max_input_length和requested_max_output_length的值分别设置为为20000和256

  ```yaml
  local_dataset_path: /data_mm/datasets/textvqa_val/textvqa_val.jsonl
  requested_max_input_length: 20000
  requested_max_output_length: 256
  ```

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

- 设置可见卡数，修改mm_run.sh文件中的ASCEND_RT_VISIBLE_DEVICES。依需求设置单卡或多卡可见。

  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1
  ```

- 运行测试命令

  ```shell
  bash scripts/mm_run.sh textvqa internvl
  ```

- 测试结果保存于以下路径。其下的results/..(一系列文件夹嵌套)/\*\_result.csv中存放着modeltest的测试结果。debug/..(一系列文件夹嵌套)/output\_\*.txt中存储着每一条数据的运行结果，第一项为output文本，第二项为输入infer函数的第一个参数的值，即模型输入。第三项为e2e_time。

  ```shell
  output/$DATE/modeltest/$MODEL_NAME/precision_result/
  ```

### VideoBench 视频+文本理解场景

- 数据准备
  - 数据集下载 [Eval_QA](https://huggingface.co/datasets/maoxx241/videobench_subset) && [Video-Bench](https://huggingface.co/datasets/LanguageBind/Video-Bench/tree/main)
  - 将`Eval_QA/`目录下的各json文件中的`vid_path`改为相应视频的绝对路径

  ```json
  ...
  "v_C7yd6yEkxXE_4": {
    "vid_path": "/data_mm/Eval_video/ActivityNet/v_C7yd6yEkxXE.mp4",
  }
  ...
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

  - 若下载有SSL相关报错，可在命令后加上 `-i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com` 参数使用阿里源进行下载
- 修改 `modeltest/config/model/internvl.yaml` 中配置项
    - 将`model_path`的值修改为模型权重的绝对路径
    - 将trust_remote_code修改为 `True` (注意：trust_remote_code为可选参数代表是否信任本地的可执行文件，默认为false。若设置为true，则信任本地可执行文件，此时transformers会执行用户权重路径下的代码文件，这些代码文件的功能的安全性需由用户保证。)

  ```yaml
  model_path: /data_mm/weights/InternVL2-8B 
  trust_remote_code: {用户输入的trust_remote_code值}
  ```

- 修改 `modeltest/config/task/videobench.yaml` 中配置项
    - 将local_dataset_path的值修改为数据集中Video-Bench-main/Eval_QA目录的绝对路径
    - (可选) 若只需运行部分子集，可将subject_mapping中不需要的子集注释掉，如下以注释ActivityNet子集为例

  ```yaml
  local_dataset_path: /data_mm/datasets/VideoBench/Video-Bench-main/Eval_QA
  ...
  subject_mapping:
  # ActivityNet:
  #  name: ActivityNet
  Driving-decision-making:
    name: Driving-decision-making
  ```

- 设置可见卡数，修改 `scripts/mm_run.sh` 文件中的`ASCEND_RT_VISIBLE_DEVICES`。依需求设置单卡或多卡可见。

  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1
  ```

- 运行测试命令

  ```shell
  bash scripts/mm_run.sh videobench internvl
  ```

- 测试结果保存于以下路径。其下的`results/..(一系列文件夹嵌套)/\*\_result.csv`中存放着modeltest的测试结果。`debug/..(一系列文件夹嵌套)/output\_\*.txt`中存储着每一条数据的运行结果，第一项为output文本，第二项为输入infer函数的第一个参数的值，即模型输入。第三项为e2e_time。

  ```shell
  output/$DATE/modeltest/$MODEL_NAME/precision_result/
  ```

## 纯模型推理场景 性能测试

_性能测试时需要在 `${image_path}` 下仅存放一张图片_

测试模型侧性能数据，需要开启环境变量

  ```shell
  export ATB_LLM_BENCHMARK_ENABLE=1
  export ATB_LLM_BENCHMARK_FILEPATH=${script_path}/benchmark.csv
  ```

**在${llm_path}目录使用以下命令运行 `run_pa.sh`**，会自动输出batchsize为1-10时，输出token长度为 256时的吞吐。

```shell
bash examples/models/internvl/run_pa.sh --performance (--trust_remote_code) ${weight_path} ${image_path}
```

可以在 `${script_path}` 路径下找到测试结果。

## 服务化推理场景 精度测试

### TextVQA 图片+文本理解场景

- 按照纯模型推理场景中TextVQA部分指导准备数据集

- 参考[服务化推理](#服务化推理)章节，在当前Docker容器中部署并启动推理服务。

- 新建同一个Docker容器的终端会话，在任意路径下使用benchmark工具进行数据集测试，执行如下benchmark命令：

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
  --ModelName internvl \
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
  --Http http://127.0.0.1:${端口号，与起服务化时config.json中的'port'保持一致} \
  --ManagementHttp http://127.0.0.2:${端口号，与起服务化时的config.json中的'managementPort'保持一致} \
  --SavePath ${日志输出路径}
  ```

完成数据集推理后，测试结果将打印展示数据集得分等指标，同时测试结果会保存在--SavePath路径下。

### VideoBench视频+文本理解场景

- 按照纯模型推理场景中VideoBench部分指导准备数据集

- 参考[服务化推理](#服务化推理)章节，在当前Docker容器中部署并启动推理服务

- 新建同一个Docker容器的终端会话，在任意路径下使用benchmark工具进行数据集测试，执行如下benchmark命令：

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
  --ModelName internvl \
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
  --Http http://127.0.0.1:${端口号，与起服务化时config.json中的'port'保持一致} \
  --ManagementHttp http://127.0.0.2:${端口号，与起服务化时的config.json中的'managementPort'保持一致} \
  --SavePath ${日志输出路径}
  ```

完成数据集推理后，测试结果将打印展示数据集得分等指标，同时测试结果会保存在--SavePath路径下。

## 服务化推理场景 性能测试

- 参考[服务化推理](#服务化推理)章节，在当前Docker容器中部署并启动推理服务

- 下载数据集[textvqa_val_performance](https://huggingface.co/datasets/maoxx241/textvqa_val_performance)

- 新建同一个Docker容器的终端会话，在任意路径下使用benchmark工具进行数据集测试，执行如下benchmark命令：

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
  --ModelName internvl \
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
  --Http http://127.0.0.1:${端口号，与起服务化时config.json中的'port'保持一致} \
  --ManagementHttp http://127.0.0.2:${端口号，与起服务化时的config.json中的'managementPort'保持一致} \
  --SavePath ${日志输出路径}
  ```

完成数据集推理后，测试结果将打印展示吞吐(GenerationSpeed)等性能指标，同时测试结果会保存在--SavePath路径下。

## FAQ

- 在对话测试或者精度测试时，用户如果需要修改输入input_texts,max_batch_size时，可以修改`${script_path}/internvl.py`里的参数，具体可见internvl.py
- 更多环境变量见[此README文件](../../README.md)

## 限制与约束

- 当用户使用多模态理解模型在服务化推理，且输入输出序列为长序列时，可能会由于超出NPU内存限制而推理失败。以 800I A2 64G 8卡，输入tokens=16384，输出tokens=2048场景为例，需要使用以下服务化参数进行配置。当用户使用的序列更长时，可以适当下调并发数。

| 模型名称 | maxPrefillTokens | npuMemSize | 最大并发数  |
|--------------|--------------|--------------|--------------|
| InternVL2.5-78B | 16384 | 20 | 35 |
