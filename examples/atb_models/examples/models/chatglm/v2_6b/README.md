# ChatGLM2-6B 模型推理指导 <!-- omit in toc -->

## 概述

- [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B/) 是开源中英双语对话模型 [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) 的第二代版本，在保留了初代模型对话流畅、部署门槛较低等众多优秀特性的基础之上，ChatGLM2-6B 有更强大的性能、更长的上下文、更高效的推理和更开放的协议。
- 此代码仓中实现了一套基于 NPU 硬件的 ChatGLM2 推理模型。配合加速库使用，旨在 NPU 上获得极致的推理性能。

## 特性矩阵

此矩阵罗列了 ChatGLM2-6B 模型支持的特性。

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8 量化 | W8A16 量化 | KV cache 量化 | 稀疏量化（仅 300I DUO 支持） | MOE | MindIE Service | TGI | 长序列 |
|-------------|-------------------------|-------------------------|------|------|-----------------|-----------------|-----------|------------|--------------|--------------------------|------|----------------|-----|--------|
| ChatGLM2-6B | 支持 world size 1,2,4,8 | 支持 world size 1,2,4 | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ |

- 此模型仓已适配的模型版本
  - [ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b/tree/main)

## 使用说明

### 路径变量解释

| 变量名      | 含义                                                                                       |
|------------|-------------------------------------------------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录（仅源码下载场景使用）                                         |
| llm_path    | 模型仓所在路径。若使用镜像，则路径为 `/usr/local/Ascend/atb-models`；若使用 gitcode 下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径；路径为 `${llm_path}/examples/models/chatglm/v2_6b`                               |
| weight_path | 模型权重路径                                                                                 |

### 版本依赖

ChatGLM2-6B 模型需使用[默认依赖](../../../../requirements/requirements.txt)版本

### 权重转换

- 参考[此 README 文件](../../../README.md)

### 量化权重导出

量化权重可通过 msmodelslim（昇腾模型压缩工具）实现。

#### 环境准备

环境配置可参考 [msmodelslim](https://gitcode.com/Ascend/msmodelslim) 官网。

#### 导出 w8a8 量化权重

通过 `${llm_path}/examples/models/chatglm/v2_6b/quant_chatglm_w8a8.sh` 文件导出模型的量化权重（注意量化权重不要和浮点权重放在同一个目录下）：

```shell
# 一定要设置该线程数
export OMP_NUM_THREADS=48
bash quant_chatglm_w8a8.sh -src ${浮点权重路径} -dst ${量化权重保存路径} -trust_remote_code
```

导出量化权重后应生成 `quant_model_weight_w8a8.safetensors` 和 `quant_model_description_w8a8.json` 两个文件。

注：

1.quant_chatglm_w8a8.sh 文件中已配置好较优的量化策略，导出量化权重时可直接使用，也可修改为其它策略。

2.执行脚本生成量化权重时，会在生成的权重路径的 config.json 文件中添加(或修改)`quantize` 字段，值为相应量化方式，当前仅支持 `w8a8`。

3.执行完以上步骤后，执行量化模型只需要替换权重路径。

4.如果生成权重时遇到 `OpenBLAS Warning: Detect OpenMP Loop and this application may hang. Please rebuild the library with USE_OPENMP = 1 option`，可通过设置 `export OMP_NUM_THREADS=1` 来关闭多线程规避。

#### 导出稀疏量化权重

> 运行前需要确保压缩工具编译过
请参考 [msmodelslim 安装指南](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/install_guide/install_guide.md) 安装 msModelSlim 量化工具。

执行 generate_sparse.sh 导出稀疏量化权重（注意量化权重不要和浮点权重放在同一个目录下）：

```shell
cd examples/atb_models/examples/models/chatglm/v2_6b/
# 导出稀疏量化权重
bash generate_sparse.sh ${浮点权重路径} ${稀疏量化权重保存路径} ${llm_path}/examples/models/chatglm/v2_6b/calib_data.jsonl ${Tensor 并行数} -trust_remote_code
```

执行后`${稀疏量化权重保存路径}`下会生成 compress 目录，使用`${稀疏量化权重保存路径}/compress` 目录作为权重目录进行推理。

注：

1.generate_sparse.sh 文件中已配置好较优的量化策略，导出量化权重时可直接使用，也可修改为其它策略。

2.执行完以上步骤后，执行量化模型只需要替换权重路径为`${稀疏量化权重保存路径}/compress`。

3.当在 npu 上生成稀疏量化权重（即`--device_type` 为 `npu` 时）时，注意需要将`${浮点权重路径}/modeling_chatglm.py` 文件 168 行的`@torch.jit.script` 注释。

### 300I DUO 运行操作说明

- 可开启 CPU Performance 模式以提高模型推理性能

  ```bash
  cpupower frequency-set -g performance
  ```

#### 对话测试

- 运行启动脚本
  - 在 `${llm_path}` 目录下执行以下指令

    ```shell
    bash ${script_path}/run_300i_duo_pa.sh ${weight_path} -trust_remote_code
    ```

- 环境变量说明
  - `export BIND_CPU=1`
    - 绑定 CPU 核心开关
    - 默认进行绑核
    - 若当前机器未设置 NUMA 或绑核失败，可将 BIND_CPU 设为 0
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`
    - 指定当前机器上可用的逻辑 NPU 核心，多个核心间使用逗号相连
    - 核心 ID 查阅方式见[此 README 文件](../../../README.md)的【启动脚本相关环境变量】章节
  - `export TP_WORLD_SIZE=2`
    - 指定模型运行时的 TP 数，即 world size
    - 默认为单卡双芯
    - 各模型支持的 TP 数参考"特性矩阵"
    - “单卡双芯”运行请指定 `TP_WORLD_SIZE` 为 `2`
  - `export MASTER_PORT=20030`
    - 设置卡间通信端口
    - 默认使用 20030 端口
    - 目的是为了避免同一台机器同时运行多个多卡模型时出现通信冲突
    - 设置时端口建议范围为：20000-20050
  - `export PYTHONPATH=${llm_path}:$PYTHONPATH`
    - 将模型仓路径加入 Python 查询模块和包的搜索路径中
    - 将 `${llm_path}` 替换为实际路径
  - 以下环境变量与性能和内存优化相关，通常情况下无需修改

    ```shell
    # 性能
    export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
    export ATB_OPERATION_EXECUTE_ASYNC=1
    export TASK_QUEUE_ENABLE=1
    export HCCL_BUFFSIZE=110
    ```

### 800I A2 运行操作说明

- 可开启 CPU Performance 模式以提高模型推理性能

  ```bash
  cpupower frequency-set -g performance
  ```

#### 对话测试

**运行 Paged Attention FP16**

- 运行启动脚本
  - 在 `${llm_path}` 目录下执行以下指令

    ```shell
    bash ${script_path}/run_800i_a2_pa.sh ${weight_path} -trust_remote_code
    ```

- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
    - 指定当前机器上可用的逻辑 NPU 核心，多个核心间使用逗号相连
    - 核心 ID 查阅方式见[此 README 文件](../../../README.md)的【启动脚本相关环境变量】章节
  - `export TP_WORLD_SIZE=1`
    - 指定模型运行时的 TP 数，即 world size
    - 默认为单卡
    - 各模型支持的 TP 数参考"特性矩阵"
  - `export MASTER_PORT=20030`
    - 设置卡间通信端口
    - 默认使用 20030 端口
    - 目的是为了避免同一台机器同时运行多个多卡模型时出现通信冲突
    - 设置时端口建议范围为：20000-20050
  - `export PYTHONPATH=${llm_path}:$PYTHONPATH`
    - 将模型仓路径加入 Python 查询模块和包的搜索路径中
    - 将 `${llm_path}` 替换为实际路径
  - `export IS_BF16=false`
    - 是否使用 BF16 精度进行推理
    - 默认使用 FP16
  - 以下环境变量与性能和内存优化相关，通常情况下无需修改

    ```shell
    # 性能
    export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
    export ATB_OPERATION_EXECUTE_ASYNC=1
    export TASK_QUEUE_ENABLE=1
    ```

**运行 Paged Attention BF16**

- 暂不支持

**运行 Paged Attention W8A8 量化**

- 运行启动脚本
  - 与"运行 Paged Attention FP16"的启动方式相同
  - `${weight_path}` 为 W8A8 量化权重的路径
- 环境变量说明
  - 参见"运行 Paged Attention FP16"中的环境变量说明
- 相比于 FP16，运行量化时需修改 W8A8 量化权重 `${weight_path}/config.json` 中的 `quantize` 字段，将此字段对应的值修改为 `w8a8`
  - 若 config.json 中无此字段，则新增

**运行 KV cache 量化**

- 暂不支持

**运行 Paged Attention 稀疏量化**

- 运行启动脚本
  - 与"运行 Paged Attention FP16"的启动方式相同
  - `${weight_path}` 为稀疏量化权重的路径
- 环境变量说明
  - 参见"运行 Paged Attention FP16"中的环境变量说明
- 相比于 FP16，运行量化时需修改稀疏量化权重 `${weight_path}/config.json` 中的 `quantize` 字段，将此字段对应的值修改为 `w8a8sc`
  - 若 config.json 中无此字段，则新增
- 注意：压缩算法与硬件强相关，当前仅 300I DUO 卡支持稀疏量化

### 精度测试

- 参考[此 README 文件](../../../../tests/modeltest/README.md)

### 性能测试

- 参考[此 README 文件](../../../../tests/modeltest/README.md)

## FAQ

- `import torch_npu` 遇到 `xxx/libgomp.so.1: cannot allocate memory in static TLS block` 报错，可通过配置 `LD_PRELOAD` 解决。
  - 示例：`export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1:$LD_PRELOAD`
