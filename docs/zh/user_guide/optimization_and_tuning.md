# 性能调优

## CPU侧优化

### 开启CPU高性能模式

``` bash
cpupower -c all frequency-set -g performance
```

### 开启透明大页

``` bash
echo always > /sys/kernel/mm/transparent_hugepage/enabled
```

### 开启jemalloc优化

jemalloc优化需要用户自行编译jemalloc动态链接库，并在脚本里引入编译好的动态链接库，具体步骤如下：

1. 下载[jemalloc源码](https://github.com/jemalloc/jemalloc)，并参考INSTALL.md文件编译安装。
2. 拉起服务前，将jemalloc动态链接库引入环境，执行如下命令：
   
``` bash
export LD_PRELOAD=${path_to_lib}/libjemalloc.so:$LD_PRELOAD
```

其中，`${path_to_lib}`为`libjemalloc.so`所在路径

## 调度特性

### 异步调度

MindIE推理的过程是同步执行，一次推理的过程按照在CPU/NPU上执行可以分为以下三个阶段：

- 数据准备阶段（CPU上执行）
- 模型推理阶段（NPU上执行）
- 数据返回阶段（CPU上执行）

异步调度的原理是使用模型推理阶段的耗时掩盖数据准备阶段和数据返回阶段的耗时，即使用NPU上执行的时间掩盖CPU上执行的时间，以及Sampling之外的CPU耗时，但是已经EOS（终止推理）的请求会被重复计算一次，造成NPU计算资源和显存资源有部分浪费。该特性适用于maxBatchSize较大，且输入输出长度比较长的场景。

设置环境变量，开启异步调度特性:

``` bash
export MINDIE_ASYNC_SCHEDULING_ENABLE=1
```

### PD分离

PD分离是指模型推理的Prefill阶段和Decode阶段分别实例化部署在不同的机器资源上同时进行推理，其结合Prefill阶段的计算密集型特性，以及Decode阶段的访存密集型特性，通过调节PD节点数量配比来提升Decode节点的batch size来充分发挥NPU卡的算力，进而提升集群整体吞吐。
此外，在Decode平均低时延约束场景，PD分离相比PD混合部署，更加能够发挥性能优势。

详细介绍和部署说明可参考[PD分离部署](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindieservice/servicedev/mindie_service0049.html)

## 并行参数

### 张量并行（TP）

张量并行（Tensor Parallel, TP）通过将张量（如权重矩阵、激活值等）在多个设备（如NPU）之间进行切分 ，从而实现模型的分布式推理。
当前所有模型默认使用张量并行方式，默认切分数为world size。

除普通的张量并行外，DeepSeek-V3、DeepSeek-R1和DeepSeek-V3.1等模型支持Lmhead矩阵local tp切分以及O project矩阵local tp切分，推荐在PD分离且D节点未分布式的场景，可以减少矩阵计算时间，降低推理时延。

| 配置项            | 取值类型 | 取值范围                 | 配置说明                                                    |
| ---------------- | ------- | ---------------------- | ---------------------------------------------------------- |
| tp               | int     | [1, worldSize]         | 张量并行数                                                  |
| lm_head_local_tp | int     | [1, worldSize / 节点数] | 表示Lmhead并行切分数。仅当tp=1时能够开启，否则默认与tp保持一致      |
| o_proj_local_tp  | int     | [1, worldSize / 节点数] | 表示Attention O矩阵切分数。仅当tp=1时能够开启，否则默认与tp保持一致 |

在800I-A3上开启Lmhead矩阵local tp切分和O project矩阵local tp切分，服务化参数配置示例如下。

``` json
{
    "ModelConfig": [
        {
            "tp": 1,
            "models": {
                "deepseekv2": {
                    "lm_head_local_tp": 16,
                    "o_proj_local_tp": 2,
                }
            }
        }
    ]
}
```

### 数据并行（DP）

数据并行（Data Parallel, DP）将推理请求划分为多个批次，并将每个批次分配给不同的设备进行并行处理，每部分设备都并行处理不同批次的数据，然后将结果合并。
数据并行支持与张量并行叠加使用，暂不支持与上下文并行叠加使用。

| 配置项 | 取值类型 | 取值范围         | 配置说明                                         |
| ----- | ------- | -------------- | ----------------------------------------------- |
| dp    | int     | [1, worldSize] | 数据并行数。与张量并行叠加时，tp * dp必须等于worldSize |

服务化参数配置参数示例如下。

``` json
{
    "ModelConfig": [
        {
            "worldSize": 8,
            "dp": 2,
            "tp": 4
        }
    ]
}
```

### 序列并行（SP）

序列并行（Sequence Parallel, SP）通过对KV Cache进行切分，使得每个sprank保存的KV Cache各不相同，达到节省显存，支持长序列的功能。
当前仅DeepSeek-V3、DeepSeek-R1、DeepSeek-V3.1等模型W8A8量化权重支持此特性。
序列并行支持与数据并行或上下文并行中任意一种并行方式叠加使用。

| 配置项 | 取值类型 | 取值范围   | 配置说明                                                      |
| ----- | ------- | -------- | ------------------------------------------------------------ |
| sp    | int     | 与tp相同  | KV Cache切分数。<br>与dp或cp叠加时，dp\*sp或cp\*sp必须等于worldSize。 |

服务化参数配置示例如下。

``` json
{
    "ModelConfig": [
        {
            "worldSize": 16,
            "dp": 2,
            "tp": 8,
            "sp": 8
        }
    ]
}
```

### 上下文并行（CP）

上下文并行（Context Parallel, CP）主要针对Self-attention模块在sequence维度进行并行计算。CP通过将长序列在上下文维度进行切分，分配到不同设备并行处理，减少首token响应时间。
当前仅DeepSeek-V3、DeepSeek-R1、DeepSeek-V3.1等模型W8A8量化权重支持此特性。
上下文并行必须与序列并行同时使用，不支持与数据并行叠加使用。

| 配置项 | 取值类型 | 取值范围 | 配置说明                                             |
| ----- | ------- | ------- | -------------------------------------------------- |
| cp    | int     | [1, 2]  | 目前开启CP特性，切分数仅支持2，且cp * tp必须等于worldSize |

服务化参数配置示例如下。

``` json
{
    "ModelConfig": [
        {
            "worldSize": 16,
            "cp": 2,
            "tp": 8,
            "sp": 8
        }
    ]
}
```

### 专家并行（EP）

MoE类模型支持专家并行（Expert Parallel, EP），通过将专家分别部署在不同的设备上，实现专家级别的并行计算。
当前实现两种形式的专家并行：

1. 基于AllGather通信的EP，即ep_level=1
2. 基于AllToAll和通算融合的EP，即ep_level=2

| 配置项    | 取值类型 | 取值范围         | 配置说明                                                |
| -------- | ------- | -------------- | ------------------------------------------------------ |
| ep_level | int     | [1, 2]         | 专家并行的实现形式                                        |
| moe_tp   | int     | [1, worldSize] | MoE部分TP切分数，默认与tp一致。当ep_level=2时，moe_tp只能为1 |
| moe_ep   | int     | [1, worldSize] | MoE部分EP切分数，moe_ep * moe_tp必须等于worldSize         |

以DeepSeek-V3为例，服务化参数配置示例如下。

``` json
{
    "ModelConfig": [
        {
            "worldSize": 16,
            "moe_tp": 1,
            "moe_ep": 16,
            "models": {
                "deepseekv2": {
                    "ep_level": 2
                }
            }
        }
    ]
}
```
