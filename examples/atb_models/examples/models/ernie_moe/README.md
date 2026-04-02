# README

- [ERNIE 4.5 MoE]是由百度推出的混合专家模型，采用创新性的多模态异构模型结构，通过跨模态参数共享机制实现模态间知识融合，同时为各单一模态保留专用参数空间。
- 此代码仓中实现了一套基于 NPU 硬件的 ERNIE 4.5 MoE 模型。

# 特性矩阵
- 矩此阵罗列了ERNIE MoE模型支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8动态量化 | W8A16量化  | KV cache量化 | 稀疏量化 | MindIE Service | TGI | 长序列   |
|-------------|----------------------------|-----------------------------|------|----------------------|-----------------|-----------------|---------|---------|----------|---------------|--------------------------|--------|-----|
| ERNIE-4.5-21B-A3B   | 支持world size 1,2,4,8     | x                  | √   | √                   | x              | √              | √   | ×          | ×           | ×                       | ×  | ×     | ×  | ×    |
| ERNIE-4.5-300B-A47B   | 支持world size 16     | ×                | √   | √                   | x              | √              | √  | ×         | ×           | ×                       | ×     | ×  | ×    |

- 注意：模型目前仅支持W8A8动态量化，不支持其它W8A8量化方法。

# 使用说明

## 路径变量解释

| 变量名      | 含义                                                                                                                                                     |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| working_dir | 加速库及模型库下载后放置的目录                                                                                                                           |
| llm_path    | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/MindIE-LLM/`；若使用 gitcode 下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径；ERNIE 4.5 MoE 的工作脚本所在路径为`${llm_path}/examples/models/ernie_moe`                                                                      |
| weight_path | 模型权重路径                                                                                                                                             |

## 权重

**权重下载**

- [ERNIE-4.5-21B-A3B-PT](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-PT)
- [ERNIE-4.5-300B-A47B-PT](https://huggingface.co/baidu/ERNIE-4.5-300B-A47B-PT)


**基础环境变量**

- 参考[此 README 文件](../../../README.md)
- 建议设置 export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3 选择workspace优化算法
- 建议设置 export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True" 选择workspace优化算法

**Chat template**
- 若要启用chat template，需要在`tokenizer_config.json`中增加`"chat_template"`字段，官方提供的chat template在`chat_template.jinja`文件中
```json
{
  "chat_template": "{%- if not add_generation_prompt is defined -%}{%- set add_generation_prompt = true -%}{%- endif -%}{%- if not cls_token is defined -%}{%- set cls_token = \"<|begin_of_sentence|>\" -%}{%- endif -%}{%- if not sep_token is defined -%}{%- set sep_token = \"<|end_of_sentence|>\" -%}{%- endif -%}{{- cls_token -}}{%- for message in messages -%}{%- if message[\"role\"] == \"user\" -%}{{- \"User: \" + message[\"content\"] + \" \" -}}{%- elif message[\"role\"] == \"assistant\" -%}{{- \"Assistant: \" + message[\"content\"] + sep_token -}}{%- elif message[\"role\"] == \"system\" -%}{{- message[\"content\"] + \" \" -}}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{- \"Assistant: \" -}}{%- endif -%}"
}
```

## 推理

### 对话测试

**运行 Paged Attention FP16**

- 运行启动脚本（transformers 版本需求：>=4.54.0）
  - 在\${llm_path}目录下执行以下指令
    ```shell
    bash ${script_path}/run_pa.sh ${weight_path}
    ```
  - trust_remote_code为可选参数代表是否信任本地的可执行文件：默认不执行。传入此参数，则信任本地可执行文件。
- 启动脚本中可设置自定义问题，具体在 input_text 后面修改即可 (默认问题为"What's Deep Learning?")
- 启动脚本中可设置自定义输出长度，具体在 max_output_length 后面修改即可（默认长度为 20）
- 若当前所用权重版本为"chat"版本，请将"--is_chat_model"赋值给 extra_param；若当前所用权重版本为"base"版本，可以将空字符串赋值给 extra_param（默认为 chat_model）
- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
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
    export ATB_CONTEXT_WORKSPACE_SIZE=1
    export ATB_LLM_ENABLE_AUTO_TRANSPOSE=0
    ```

## 精度测试

- 参考[此 README 文件](../../../tests/modeltest/README.md)
  - 示例 
    ```shell
    cd ${llm_path}/tests/modeltest
    bash run.sh pa_fp16 full_BoolQ 1 ernie_moe ${ERNIE-4.5-21B-A3B权重路径} 4
    ```

## 性能测试

- 参考[此 README 文件](../../../tests/modeltest/README.md)
  - 示例
    ```shell
    cd ${llm_path}/tests/modeltest
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 ernie_moe ${ERNIE-4.5-21B-A3B权重路径} 4
    ```

## FAQ

- 更多环境变量见[此 README 文件](../../README.md)
- 对话测试实际执行的 Python 文件为`${llm_path}/examples/run_pa.py`；这个文件的参数说明见[此 README 文件](../../README.md)