# README

- [DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2)是杭州深度求索人工智能基础技术研究有限公司发布的专家混合（MoE）语言模型，其特点是训练经济，推理高效。其主要创新点是：（1）推出了MLA (Multi-head Latent Attention)，其利用低秩键值联合压缩来消除推理时键值缓存的瓶颈，从而支持高效推理；（2）在FFN部分采用了DeepSeekMoE架构，能够以更低的成本训练更强的模型。

- 此代码仓中实现了一套基于NPU硬件的DeepSeek-V2推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

# 特性矩阵
- 此矩阵罗列了DeepSeek-V2模型支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16（仅800I A2支持） | Flash Attention | Paged Attention | W8A8量化 | W8A16量化  |KV cache量化 | 稀疏量化（仅300I DUO支持） | MindIE Service | TGI | 长序列  |
|-------------|----------------------------|-----------------------------|------|----------------------|-----------------|-----------------|---------|-----------|-----------|--------------|--------------------------|--------|-----|
| DeepSeek-V2-Chat-236B    | 支持world size 16     | ×                | √   | √                   | √              | √              | √       | √              | ×           | ×                       | √     | ×  | ×  |


## 路径变量解释

| 变量名      | 含义                                                                                                                                                     |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| working_dir     | 加速库及模型库下载后放置的目录                                                                                                                           |
| llm_path        | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/MindIE-LLM/`；若使用 gitcode 下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path     | 脚本所在路径；Deepseek-MoE 的工作脚本所在路径为`${llm_path}/examples/models/deepseekv2`                                                                    |
| weight_path     | 模型权重路径                                                                                                                                             |
| rank_table_path | Rank table文件路径                                                                                                                                              |

## 权重

**权重下载**

- [Deepseek-V2-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat)


**基础环境变量**

- 参考[此 README 文件](../../../README.md)
- 建议设置 export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3 选择workspace优化算法
- 建议设置 export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True" 选择workspace优化算法
- 建议设置 export HCCL_BUFFSIZE=200 若CANN报错日志出现“HCCL_BUFFSIZE is too SMALL”字样，请根据日志中的公式并向上取整设置合理的HCCL_BUFFSIZE，此处建议公式：batch_size * hidden_size * 2 * 2 * localMoeExpertNum，向上取整。用于控制两个NPU之间共享数据的缓存区大小。单位为M，需要配置为整数，取值大于等于1，默认值200

## 生成量化权重

- 生成量化权重依赖msModelSlim工具，安装方式见[此README](https://gitcode.com/ascend/msit/tree/dev/msmodelslim)。
- 量化权重统一使用`${llm_path}/examples/convert/model_slim/quantifier.py`脚本生成，以下提供DeepSeek-V2模型量化权重生成快速启动命令，各模型量化方式的具体参数配置见`${llm_path}/examples/models/deepseekv2/generate_quant_weight.sh`
- 当前DeepSeek-V2支持W8A16、W8A8 dynamic量化，通过以下命令生成量化权重：
```shell
# 设置CANN包的环境变量
source /usr/local/Ascend/cann/set_env.sh
cd ${llm_path}
# 生成w8a16量化权重
bash examples/models/deepseekv2/generate_quant_weight.sh -src {浮点权重路径} -dst {量化权重路径} -type deepseekv2_w8a16 -trust_remote_code
# 生成w8a8 dynamic量化权重
bash examples/models/deepseekv2/generate_quant_weight.sh -src {浮点权重路径} -dst {量化权重路径} -type deepseekv2_w8a8_dynamic -trust_remote_code

```
- **MLA W8A16 + MoE W8A8 Dynamic混合精度量化**：生成w8a8 dynamic量化权重后，进行如下操作：
  - 修改`config.py`文件，新增`"mla_quantize": "w8a16"`
  - 修改`quant_model_description_w8a8_dynamic.json`文件，将包含`self_attn`的字段中`W8A8_DYNAMIC`修改为`W8A16`

- 若启用动态负载均衡特性场景，需要将量化后的权重再执行一次权重文件NZ转换
```shell
# 设置CANN包的环境变量
source /usr/local/Ascend/cann/set_env.sh
# 设置模型仓环境变量
# 若使用编译好的包，则执行以下指令
source ${llm_path}/set_env.sh
# 若使用gitcode上的源码进行编译，则执行以下指令
source ${llm_path}/output/atb_models/set_env.sh
cd ${llm_path}
# 创建转NZ后的存储权重路径
mkdir -p {转NZ后权重路径}
# 因为动态负载均衡会加载磁盘中的权重文件，建议根据需要提前创建内存文件系统，这样可以提升加载到host侧的性能，如下命令：
mount tmpfs {转NZ后权重路径} -o huge=always -t tmpfs

bash examples/models/deepseekv2/cast_weight_to_nz.sh -src {原量化权重路径} -dst {转NZ后权重路径} -index {权重文件的index.json文件}
```

## 推理

执行推理前请修改权重文件夹的`config.json`文件：

- 修改`model_type`字段为`"deepseekv2"`

### 对话测试
**运行Paged Attention FP16**
- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
    - 指定当前机器上可用的逻辑NPU核心，多个核心间使用逗号相连
    - 核心ID查阅方式见[此README文件](../../README.md)的【启动脚本相关环境变量】章节
  - `export MASTER_PORT=20030`
    - 设置卡间通信端口
    - 默认使用20030端口
    - 目的是为了避免同一台机器同时运行多个多卡模型时出现通信冲突
    - 设置时端口建议范围为：20000-20050
  - 以下环境变量与性能和内存优化相关，通常情况下无需修改
    ```shell
    export INF_NAN_MODE_ENABLE=0
    export ATB_LLM_ENABLE_AUTO_TRANSPOSE=0
    ```
- 运行启动脚本
  - 在\${llm_path}目录下执行以下指令
    ```shell
    bash ${script_path}/run_pa.sh ${weight_path} -trust_remote_code
    ```
  - trust_remote_code为可选参数代表是否信任本地的可执行文件：默认不执行。传入此参数，则信任本地可执行文件。
- 运行attention data parallel
  - 在\${llm_path}目录下执行以下指令
    ```shell
    bash ${script_path}/run_pa.sh ${weight_path} ${dp} ${tp} ${sp} ${moe_tp} ${moe_ep}
    ```
  - 并行参数说明
    - `dp`为数据并行数，`tp`为张量并行数，`sp`为序列并行数，`moe_tp`为MoE张量并行数，`moe_ep`为MoE专家并行数
    - 需满足`dp` * `tp` = `world_size`（总卡数），`moe_ep` * `moe_tp` = `world_size`，`sp` = `tp`
  - 示例
    ```shell
    bash ${script_path}/run_pa.sh ${weight_path} 8 1 1 1 8
    ```

## 精度测试

- 参考[此 README 文件](../../../tests/modeltest/README.md)
  - 单机示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export ATB_LLM_ENABLE_AUTO_TRANSPOSE=0
    bash run.sh pa_bf16 full_BoolQ 1 deepseekv2 ${weight_path} 8
    bash run.sh pa_bf16 full_CEval 5 1 deepseekv2 ${weight_path} 8
    ```
  - 双机示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export ATB_LLM_BENCHMARK_ENABLE=1
    export ATB_LLM_ENABLE_AUTO_TRANSPOSE=0

    # 以下两条命令需要在两个节点同步执行
    # 节点1
    bash run.sh pa_bf16 full_BoolQ 1 deepseekv2 ${weight_path} ${rank_table_path} 16 2 0 [master_address]
    # 节点2
    bash run.sh pa_bf16 full_BoolQ 1 deepseekv2 ${weight_path} ${rank_table_path} 16 2 8 [master_address]
    ```
  - attention data parallel示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export ATB_LLM_ENABLE_AUTO_TRANSPOSE=0
    # bash run.sh pa_[data_type] [dataset] ([shots]) [batch_size] [model_name] [weight_dir] [world_size] [dp,tp,sp,moe_tp,moe_ep]
    bash run.sh pa_bf16 full_BoolQ 16 deepseekv2 ${weight_path} 8 [8,1,1,1,8]
    bash run.sh pa_bf16 full_CEval 5 16 deepseekv2 ${weight_path} 8 [8,1,1,1,8]
    ```

## 性能测试

- 参考[此 README 文件](../../../tests/modeltest/README.md)
  - 单机示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export ATB_LLM_BENCHMARK_ENABLE=1
    export ATB_LLM_ENABLE_AUTO_TRANSPOSE=0
    bash run.sh pa_bf16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 deepseekv2 ${weight_path} 8
    ```
  - 双机示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export HCCL_OP_EXPANSION_MODE="AIV"
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export ATB_LLM_BENCHMARK_ENABLE=1
    export ATB_LLM_ENABLE_AUTO_TRANSPOSE=0

    # 以下两条命令需要在两个节点同步执行
    # 节点1
    bash run.sh pa_bf16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 deepseekv2 ${weight_path}
    ${rank_table_path} 16 2 0 [master_address]
    # 节点2
    bash run.sh pa_bf16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 deepseekv2 ${weight_path}
    ${rank_table_path} 16 2 8 [master_address]
    ```
  - attention data parallel示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export ATB_LLM_BENCHMARK_ENABLE=1
    export ATB_LLM_ENABLE_AUTO_TRANSPOSE=0
    # bash run.sh pa_[data_type] performance [case_pair] [batch_size] [model_name] [weight_dir] [world_size] [dp,tp,sp,moe_tp,moe_ep]
    bash run.sh pa_bf16 performance [[1,512]] 512 deepseekv2 ${weight_path} 8 [8,1,1,1,8]
    ```

## FAQ

- 更多环境变量见[此 README 文件](../../README.md)
- 对话测试实际执行的 Python 文件为`${llm_path}/examples/run_pa.py`；这个文件的参数说明见[此 README 文件](../../README.md)