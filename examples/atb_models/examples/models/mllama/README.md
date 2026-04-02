# README

- [LlaMa 3.2](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD_VISION.md)多模态大语言模型系列，是 Meta AI 发布的能够支持文本+图像输入/图像输出的预训练大模型集合，可以通过图像以及自然语言交互的方式提供知识、文本生成、语言翻译、语言理解、代码编写、视觉识别、图像推理、字幕和解释等任务。

- 此代码仓中实现了一套基于NPU硬件的LlaMa多模态推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

# 特性矩阵
- 此矩阵罗列了各LlaMa多模态模型支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16（仅800I A2支持） | Flash Attention | Paged Attention | W8A8量化 | W8A16量化 | KV cache量化 | 稀疏量化（仅300I DUO支持） | MOE | MindIE Service | TGI | 长序列 |
|-------------|----------------------------|-----------------------------|------|----------------------|-----------------|-----------------|----------|----------|--------------|--------------------------|-----|----------------|-----|--------|
| LlaMa3.2-11B| 支持world size 1,2,4,8     | ×                           | ×    | √                    | ×               | √               | ×        | ×        | ×            | ×                        | ×   | ×              | ×   |×       |
| LlaMa3.2-90B| 支持world size 4,8         | ×                           | ×    | √                    | ×               | √               | ×        | ×        | ×            | ×                        | ×   | ×              | ×   |×       |

- 此模型仓已适配的模型版本
  - [LlaMa多模态系列](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD_VISION.md)

# 使用说明

## 路径变量解释
| 变量名  | 含义                                             |
|--------|--------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                 |
| llm_path | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/llm_model/`；若使用gitcode下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models`    |
| weight_path | 模型权重路径                                |
| image_path  | 图片路径                                   |
| max_batch_size  | 最大bacth数                            |
| max_input_length  | 多模态模型的最大embedding长度，        |
| max_output_length | 生成的最大token数                     |

## 权重
### 权重下载
- [LlaMa3.2-11B](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct/tree/main)
- [LlaMa3.2-90B](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct/tree/main)

### 权重转换
- 参考[此README文件的【权重转换】章节](../../README.md)

## 推理
> **说明：**
> 运行时请确认权重`${weight_path}/config.json`中的`torch_dtype`、`kv_quant`和`quantize`字段配置正确，参考[此README文件的【权重设置】章节](../../README.md)

### 对话测试
**运行Paged Attention**
- 运行启动脚本
  - 在\${llm_path}目录下执行以下指令
    ```shell
    bash ${script_path}/run_pa.sh --run ${weight_path} ${image_path} ${max_batch_size} ${max_input_length} ${max_output_length}
    ```
- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
    - 指定当前机器上可用的逻辑NPU核心，多个核心间使用逗号相连
    - 核心ID查阅方式见[此README文件](../../README.md)的【启动脚本相关环境变量】章节
    - 对于300I DUO卡而言，若要使用单卡双芯，请指定至少两个可见核心；若要使用双卡四芯，请指定至少四个可见核心
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
  - 以下环境变量开启确定性计算，在精度测试以及对性能要求不严格场景下开启
   ```shell
    export LCCL_DETERMINISTIC=1
    export HCCL_DETERMINISTIC=true
    export ATB_MATMUL_SHUFFLE_K_ENABLE=0
    export ATB_LLM_LCOC_ENABLE=0
   ```  

## FAQ
- 更多环境变量见[此README文件的【部分环境变量介绍】章节](../../README.md)
- 对话测试实际执行的Python文件为`${llm_path}/examples/run_fa.py`和`${llm_path}/examples/run_pa.py`；这两个文件的参数说明见[此README文件的【启动脚本】章节](../../README.md)