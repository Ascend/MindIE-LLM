# README

- [LLaVA（Large Language and Vision Assistant）](https://github.com/haotian-liu/LLaVA)，是一种多模态大模型，具有强大的图像和文本处理能力，使得它在聊天机器人等场景中具有广泛的应用前景。 在聊天机器人中，LLaVA 可以通过解析用户的文字输入，结合图像信息，生成更加生动、准确的回复。 此外，LLaVA 还可以根据用户的图像输入，提供相关的文本信息，实现更加智能化的交互。
- 此代码仓中实现了一套基于 NPU 硬件的 LLaVa 推理模型。配合加速库使用，旨在 NPU 上获得极致的推理性能。
- 支持 llava llava-next 和 llava-next-video 等系列模型的多模态推理

## 特性矩阵

| 模型及参数量      | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | 800I A2 BF16 | MindIE Service |纯模型支持模态  | 服务化支持模态 |
|-------------|----------------------------|-----------------------------|------|------------------|-----------------|-----|-----|
|  llava1.5-7B    | 支持 world size 1,2,4,8     | 支持 world size 1,2,4,8        | ✅   |  ✅                   | ✅              | 文本、图片           | 文本、图片|
|  llava1.5-13B    | 支持 world size 1,2,4,8     | 支持 world size 1,2,4,8        | ✅   |  ✅                   | ✅              | 文本、图片           | 文本、图片|
|  llava1.6-7B    | 支持 world size 1,2,4,8     | 支持 world size 1,2,4,8        | ✅   |  ✅                   | ✅              | 文本、图片           | 文本、图片|
|  llava1.6-13B    | 支持 world size 1,2,4,8     | 支持 world size 1,2,4,8        | ✅   |  ✅                   | ✅              | 文本、图片           | 文本、图片|
|  llava1.6-34B    | 支持 world size 1,2,4,8     | 支持 world size 1,2,4,8        | ✅   |  ✅                   | ✅              | 文本、图片           | 文本、图片|
|  llava-next-video-7B    | 支持 world size 1,2,4,8     | 支持 world size 1,2,4,8        | ✅   |  ✅                   | ✅              | 文本、图片、视频           | 文本、图片|
|  llava-next-video-34B    | 支持 world size 1,2,4,8     | 支持 world size 1,2,4,8        | ✅   |  ✅                   | ✅              | 文本、图片、视频           |文本、图片|

须知：

1. 当前版本服务化仅支持单个请求单张图片输入
2. 当前多模态场景，MindIE Service 仅支持 MindIE Service、TGI、Triton、vLLM Generate 4 种服务化请求格式

## 路径变量解释

| 变量名      | 含义                                                                                                                                                         |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| working_dir | 加速库及模型库下载后放置的目录                                                                                                                               |
| llm_path    | 模型仓所在路径。若使用镜像，则路径为 `/usr/local/Ascend/atb-models`；若使用 gitcode 下载的代码，则路径为 `{working_dir}/MindIE-LLM/examples/atb_models`（仅源码下载场景使用） |
| script_path | 脚本所在路径；llava 的工作脚本所在路径为 `${llm_path}/examples/models/llava`                                                                                          |
| weight_path | 模型权重路径                                                                     |
| image_path  | 图片所在路径                                                                     |
| trust_remote_code  | 是否信任本地的可执行文件：默认不执行，传入此参数，则信任                      |
| max_batch_size  | 最大 batch 数                                                                  |
| video_frames | 输入为 video 时，抽取的帧数                                                        |
| max_input_length  | 多模态模型的最大 embedding 长度，                                             |
| max_output_length | 生成的最大 token 数                                                          |
|open_clip_path| open_clip 权重所在路径                                                           |
|llava_type |  当前 llava 模型类型，可选为 llava, llava_next 和 llava_next_video                       |

-注意：
max_input_length 长度设置可参考模型权重路径下 config.json 里的 max_position_embeddings 参数值
如果是在 Atlas 800I A2 硬件上跑 13B 模型，可适当减小该值防止出现 OOM 错误

## 权重

**权重下载**

- [LLava-1.5-7B](https://huggingface.co/llava-hf/llava-1.5-7b-hf/tree/main)
- [LLava-1.5-13B](https://huggingface.co/llava-hf/llava-1.5-13b-hf/tree/main)
- [LLava-1.6-mistral-7B](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/tree/main)
- [LLava-1.6-vicuna-7B](https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf/tree/main)
- [LLava-1.6-vicuna-13B](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf/tree/main)
- [LLava-1.6-34B](https://huggingface.co/llava-hf/llava-v1.6-34b-hf/tree/main)
- [LLava-1.6-video-7B](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf/tree/main)
- [LLava-1.6-video-34B](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-34B-hf/tree/main)

**基础环境变量**

- Python 其他第三方库依赖，参考 [requirements_llava.txt](../../../requirements/models/requirements_llava.txt)

- 参考[此 README 文件](../../../README.md)
-注意：保证先后顺序，否则 llava 的其余三方依赖会重新安装 torch，导致出现别的错误
 llava-next-video 需要再安装 av 三方件 pip install av,且 transformers 版本大于等于 4.42.0

**量化权重生成**

- 基于原始的浮点权重，使用量化工具，将高位浮点数转为低位的定点数。
- 生成量化权重依赖 msModelSlim 工具，安装方式见[此 README](https://gitcode.com/Ascend/msmodelslim)。
- 注意事项：
  - `model_path` 和 `save_directory` 请勿使用同一个文件夹，避免浮点权重和量化权重混淆
  - NPU 多卡量化注意事项和环境要求见[此 README 中的【NPU 多卡量化】章节](../../README.md)
  - 当前仅支持对 LLM 模型的 W8A16 量化，暂不支持 VIT 模型的量化

    ```shell
    # 设置 CANN 包的环境变量
    source /usr/local/Ascend/cann/set_env.sh
    cd ${llm_path}
    bash examples/models/llava/convert_quant_weights.sh -src {浮点权重路径} -dst {W8A16 量化权重路径} -type llava_w8a16
    ```

## 推理

### 对话测试

**运行 Paged Attention FP16**

- 运行启动脚本
  - 在\${llm_path} 目录下执行以下指令

    ```shell
    bash ${script_path}/run_pa.sh --run (--trust_remote_code) ${weight_path} ${image_path} ${video_frames} ${max_batch_size} ${max_input_length} ${max_output_length}
    ```

- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
    - 指定当前机器上可用的逻辑 NPU 核心，多个核心间使用逗号相连
    - 核心 ID 查阅方式见[此 README 文件](../../README.md)的【启动脚本相关环境变量】章节
    - 对于 300I DUO 卡而言，若要使用单卡双芯，请指定至少两个可见核心；若要使用双卡四芯，请指定至少四个可见核心
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
    export ATB_CONTEXT_WORKSPACE_SIZE=0
    ```

## 精度测试

### 方案

我们采用的精度测试方案是这样的：使用同样的一组图片，分别在 GPU 和 NPU 上执行推理，得到两组图片描述。 再使用 open_clip 模型作为裁判，对两组结果分别进行评分，以判断优劣。

#### 实施

1. 下载 [open_clip 的权重 open_clip_pytorch_model.bin](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main)，并把下载的权重放在 open_clip_path 目录下
   下载[测试图片（CoCotest 数据集）](https://cocodataset.org/#download)并随机抽取其中 100 张图片放入{image_path} 目录下
   - 安装 open_clip 仓库（众多 github 下载的库可以参照如下方式，快速安装）

    ```shell
   # 在命令行界面中手动克隆 open_clip 仓库，进入克隆下来的 open_clip 目录 pip 安装
    git clone https://github.com/mlfoundations/open_clip.git
    cd open_clip
    pip install -e .


2. GPU 上，在{script_path}/precision 目录下，运行脚本 python run_coco.py --model_path ${weight_path} --image_path ${image_path} --lava_type ${llava_type},会在{script_path}/precision 目录下生成 coco_predict.json 文件存储 gpu 推理结果

3. NPU 上,在\${llm_path} 目录下执行以下指令：

   ```bash
   bash ${script_path}/run_pa.sh --precision (--trust_remote_code) ${weight_path} ${image_path} ${max_batch_size} ${max_input_length} ${max_output_length}
   ```

   运行完成后会在{script_path} 生成 predict_result.json 文件存储 npu 的推理结果

4. 对结果进行评分：分别使用 GPU 和 NPU 推理得到的两组图片描述(coco_predict.json、predict_result.json)作为输入,执行 clip_score_llava.py 脚本输出评分结果

```bash
   python examples/models/llava/precision/clip_score_llava.py \
   --model_weights_path {open_clip_path}/open_clip_pytorch_model.bin \
   --image_info {coco_predict.json 或 predict_result.json 的路径} \
   --dataset_path {iamge_path}
```

   得分高者精度更优。

## 性能测试

性能测试时需要在 `${image_path}` 下仅存放一张图片，使用以下命令运行 `run_pa.sh`，会自动输出 batchsize 为 1-10 时，输出 token 长度为 256 时的吞吐。Atlas 800I A2 上硬件只能跑单 batch，如果需要多跑 batch，可以尝试用多张卡跑。

```shell
bash ${script_path}/run_pa.sh --performance (--trust_remote_code) ${weight_path} ${image_path} ${video_frames} ${max_batch_size} ${max_input_length} ${max_output_length}
```

测试性能时，需要导入环境变量:export ATB_LLM_BENCHMARK_ENABLE=1，export ATB_LLM_BENCHMARK_FILEPATH=${script_path}/benchmark.csv

例如在 MindIE-ATB-Models 根目录，可以运行：

```shell
bash examples/models/llava/run_pa.sh --performance (--trust_remote_code) ${weight_path} ${image_path} ${video_frames} ${max_batch_size} ${max_input_length} ${max_output_length}
```

可以在 `examples/models/llava` 路径下找到测试结果。

## FAQ

- 在对话测试或者精度测试时，用户如果需要修改输入 input_texts,max_batch_size 时，可以修改{script_path}/llava.py 里的参数，具体可见 llava.py
- 更多环境变量见[此 README 文件](../../README.md)
