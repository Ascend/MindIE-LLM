# DeepSeek-V3.2 模型部署指南

## 简介

- [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) 是由深度求索（DeepSeek）公司发布的 MoE 架构大语言模型，通过引入 DeepSeek 稀疏注意力（DeepSeek Sparse Attention, DSA）机制，重新定义了大模型效率标准，可通过自然语言交互的方式完成文本生成、代码编写和解释、数学推理等任务。
- DeepSeek-V3.2 具备卓越的推理能力，该模型使用 DSA 稀疏注意力机制将注意力计算复杂度由 $O(L^2)$ 降至 $O(Lk)$，显著降低计算量和显存占用，在保持基准性能的同时大幅提升了推理效率。其中，$L$ 为序列长度，$k (k \ll L)$ 为选择的 token 数量。
- MindIE-LLM 在 NPU 上实现了 DeepSeek-V3.2 的高性能推理，采用 aclgraph 图模式方案，旨在获得极致的推理性能。

---

## 特性矩阵

**表1** 硬件支持

|模型|Atlas 800I A2|Atlas 800I A3|Atlas 300I Duo 推理卡|
|:-----:|:--------:|:-----------:|:------------------:|
|DeepSeek-V3.2|四机 32 卡部署|双机 16 卡部署|❌|

**表2** 浮点和量化

|     模型      | W8A8 量化 | 稀疏量化 | W4A8 量化 | KV cache 量化 | FA3 量化 |
| :-----------: | :-------: | :------: | :-------: | :-----------: | :------: |
| DeepSeek-V3.2 |     ✅     |    ❌     |     ❌     |       ❌       |    ❌     |

**表3** 其他特性

| 模型 | Multi-Lora | 负载均衡 | Data Parallel | Context Parallel | Tensor Parallel | Expert Parallel | Flash Comm v1 | 异步调度 | Chunked Prefill | SLO 调度优化 | Micro Batch | MTP | Prefix Cache | KV Cache 池化 | Function Call | 思考解析 | PD 分离 |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----:  | :-----: |
| DeepSeek-V3.2 | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌  | ✅ | ✅ | ✅ |

**表4** 特性叠加

| 特性 | Data Parallel | Context Parallel | W8A8 量化 | 异步调度 | Chunked Prefill | MTP  | Prefix Cache | Function Call | 思考解析 | PD 分离 |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| Data Parallel    | ✅ |
| Context Parallel | ❌ | ✅ |
| W8A8 量化        | ✅ | ✅ | ✅ |
| 异步调度          | ✅ | ❌ | ✅ | ✅ |
| Chunked Prefill  | ✅ | ❌ | ✅ | ✅ | ✅ |
| MTP             | ✅ | ✅ | ✅ | ✅ | 混部❌<br/>PD分离✅ | ✅ |
| Prefix Cache    | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Function Call   | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 思考解析          | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| PD 分离          | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 环境准备

### 权重量化

可以直接在 Modelers 上下载 DeepSeek-V3.2 模型的 W8A8 量化权重，或者使用 msmodelslim 量化工具基于浮点权重进行量化。

#### 昇腾原生量化权重下载

可以通过 Modelers 开源社区直接下载昇腾原生 W8A8 量化权重：

- [DeepSeek-V3.2-w8a8-mtp-QuaRot](https://modelers.cn/models/Eco-Tech/DeepSeek-V3.2-w8a8-mtp-QuaRot)

#### 使用 msmodelslim 工具生成量化权重

可以使用 [msmodelslim](https://gitcode.com/Ascend/msmodelslim) 生成量化权重，请参考[一键量化使用指南](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/feature_guide/quick_quantization_v1/usage.md)进行配置并完成量化权重的生成，下面给出量化方式简介和相对应的生成命令。

**量化策略说明：**

DeepSeek-V3.2 的 W8A8 量化采用多阶段流水线，依次执行以下步骤：

1. **QuaRot（旋转量化）**：通过数学旋转变换消除激活值中的离群点，为后续量化提供更友好的数值分布。
2. **Flex Smooth Quant（离群值平滑）**：对旋转后的模型进一步平滑离群值，覆盖 `norm-linear` 和 `ov`（共享专家）子图结构。
3. **线性层量化**：对 Attention 和 MoE 模块分别采用不同的量化策略：

| 模块 | 量化方式 | 激活值量化 | 权重量化 | 说明 |
|------|----------|-----------|---------|------|
| Attention（self_attn） | W8A8 静态量化 | per_tensor, int8, 非对称 | per_channel, int8, 对称 | 排除 kv_b_proj、wq_b、wk、weights_proj |
| MLP/MoE 专家 | W8A8 动态量化 | per_token, int8, 对称 | per_channel, int8, 对称 | 排除 gate 层 |

**量化命令示例：**

生成 DeepSeek-V3.2 W8A8-QuaRot 量化权重：

```shell
msmodelslim quant \
  --model_path ${MODEL_PATH} \     # 模型权重路径
  --save_path ${SAVE_PATH} \       # 量化权重保存路径
  --model_type DeepSeek-V3.2 \     # 必须为 DeepSeek-V3.2
  --quant_type w8a8 \
  --trust_remote_code True
```

### 软件环境

1. 镜像/物理机/容器安装场景所需的环境准备，请参见[环境准备](../../install/environment_preparation.md)。
2. 如果是多机场景，需要参考 [Rank Table File 配置指南](../../user_manual/rank_table_file_configuration.md) 配置 rank\_table\_file.json 文件。

- rank\_table\_file.json 配置完成后，需要执行命令修改权限为 640

- 如果以普通用户权限部署服务化（例如，用户名为 HwHiAiUser，对应的用户 ID 通常为 1001），需要修改模型目录及目录下文件属主为 1001（root 用户权限可忽略），修改权重目录权限为 750：

```shell
chown -R 1001:1001 {/path-to-weights/DeepSeek-V3.2}
chmod 750 {/path-to-weights/DeepSeek-V3.2}
```

---

## 安装

MindIE-LLM 安装请参见[安装指南](../../install/installing_MindIE.md)。

---

## 服务化推理

### Atlas 800I A3 双机部署

#### 配置服务化环境变量

在 2 台机器上都需要按照下述方法设置环境变量：

```shell
source /usr/local/lib/python3.11/site-packages/mindie_llm/set_env.sh # 设置 MindIE-LLM 运行所需环境变量
export MINDIE_LOG_TO_STDOUT=1                            # 开启日志输出到屏幕（可选）
export TASK_QUEUE_ENABLE=0                               # 关闭任务队列，避免多流场景下精度问题
export HCCL_OP_EXPANSION_MODE="HOST"                     # 使用 HOST 模式，避免通信算子偶发报错

# 显存优化
export NPU_MEMORY_FRACTION=0.92                          # 设置 NPU 显存比例
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True" # 配置 Pytorch 显存可扩展分段

# 性能优化
export HCCL_ALGO="level0:NA;level1:pipeline"             # 设置 HCCL 算法
export MINDIE_ASYNC_SCHEDULING_ENABLE=1                  # 开启异步推理（可选但推荐，性能极致调优依赖异步推理，HOST 和 DEVICE 时延互相掩盖）

# 多机推理相关环境变量，需要根据实际环境情况进行设置
export HCCL_CONNECT_TIMEOUT=7200                         # 设置 HCCL 连接超时时间，避免连接超时导致服务拉起失败
export RANK_TABLE_FILE="/path/to/rank_table_file.json"   # 在环境准备章节中预先配置好的 rank_table_file.json 文件路径
export MASTER_IP=xxx.xxx.xxx.xxx                         # 主节点 IP
export MIES_CONTAINER_IP=xxx.xxx.xxx.xxx                 # 本机 IP
export MASTER_PORT=xxxx                                  # 设置主机端口号，可选范围[0, 65535]，且不和本机上其他服务的端口冲突即可
export GLOO_SOCKET_IFNAME=xxxx                           # 设置 GLOO 通信网卡

# 如果容器内设置了代理，则需要取消设置，避免双机通信出现异常
unset http_proxy https_proxy
```

说明：

- `MIES_CONTAINER_IP` 优先级高于配置文件中的 `ipAddress`，设置后发送请求时，以主节点的 `MIES_CONTAINER_IP` 为准。
- `GLOO_SOCKET_IFNAME` 需要根据 MIES_CONTAINER_IP 对应的网卡名，可以通过 `ifconfig` 查看网卡列表，设置为对应的网卡名，参考[FAQ](https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/faq/faq.md#gloo%E8%BF%9E%E6%8E%A5%E5%A4%B1%E8%B4%A5%E6%8A%A5%E9%94%99%EF%BC%9Aerror-failed-to-connect-errorso_error-connection-refused)。

#### 配置服务化参数

在 2 台机器上都需要按照下述方法修改服务化参数，进入 MindIE-LLM 安装目录，编辑服务化配置文件：

```shell
cd /usr/local/lib/python3.11/site-packages/mindie_llm
vim conf/config.json
```

修改以下参数：

```json
{
    "ServerConfig" :
    {
        "httpsEnabled" : false, # 关闭 HTTPS 后，客户端与服务端之间的请求将以明文传输，建议仅在安全内网中关闭
        ...
    },

    "BackendConfig" : {
        "npuDeviceIds" : [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]], # 当前机器可使用的 NPU 卡，对于A3，每卡有2带，8卡共16带
        "multiNodesInferEnabled" : true,     # 开启多机推理
        "interNodeTLSEnabled" : false,       # 关闭跨机 TLS 后节点间的通信数据将以明文传输，建议仅在安全内网中关闭
        ...
        "ModelDeployConfig" :
        {
            ...
            "ModelConfig" : [
                {
                    ...
                    "modelName" : "DeepSeek-V3.2",                                      # 模型名，不影响服务化拉起
                    "modelWeightPath" : "/mnt/weights/DeepSeek-V3.2-w8a8-mtp-QuaRot",   # 权重路径
                    "worldSize" : 16,
                    "cpuMemSize" : 0,
                    "npuMemSize" : -1,
                    "backendType" : "torch",  # 选择推理框架后端，DeepSeek-V3.2 必须选择 torch
                    "dp": 4,                  # 数据并行
                    "tp": 8,                  # 张量并行
                    "cp": 1,                  # 上下文并行
                    "sp": 1,                  # 序列并行
                    "pp": 1,                  # pipeline 并行
                    "moe_ep": 32,             # MOE 专家并行
                    "moe_tp": 1,              # MOE 张量并行
                    "plugin_params": "{\"plugin_type\":\"mtp\",\"num_speculative_tokens\":2}" # 开启 MTP 特性，且投机 token 个数为 2
                    ...
                }
            ]
        },
        ...
    },
    ...
}
```

`maxSeqLen`、`maxInputTokenLen`、`maxPrefillBatchSize`、`maxPrefillTokens`、`maxBatchSize` 和 `maxIterTimes` 均需要根据实际场景进行配置，建议根据模型大小和卡数进行调整，以获得最佳性能。

#### 拉起服务

```shell
# 拉起服务化，在 2 台环境上都需要执行
mindie_llm_server
```

执行命令后，首先会打印本次启动所用的所有参数，然后直到 2 台环境上都出现以下输出，表示服务拉起成功：

```shell
Daemon start success!
```

#### 功能验证

在服务拉起成功后，重新打开一个终端窗口，使用以下命令发送请求，其中 `{MASTER_IP}` 需要改为 `MASTER_IP`，`{PORT}` 需要改为服务化 `config.json` 文件中配置的 port。

```shell
curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{
  "model": "DeepSeek-V3.2",
  "max_tokens": 20,
  "messages": [{"role": "user", "content": "What is deep learning?"}]
}' http://{MASTER_IP}:{PORT}/v1/chat/completions
```

回显如下，表明请求发送并推理成功：

```log
{"id":"endpoint_common_1","object":"chat.completion","created":1774785542,"model":"DeepSeek-V3.2","choices":[{"index":0,"message":{"role":"assistant","content":"Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to","tool_calls":[]},"logprobs":null,"finish_reason":"length"}],"usage":{"prompt_tokens":9,"prompt_tokens_details":{"cached_tokens":0},"completion_tokens":20,"completion_tokens_details":{"reasoning_tokens":0},"total_tokens":29}}
```

### Atlas 800I A2 八机大 EP 部署

参考 [MindIE-Motor](https://gitcode.com/Ascend/MindIE-Motor/blob/master/docs/zh/user_guide/service_deployment/pd_separation_service_deployment.md#%E5%AE%89%E8%A3%85%E9%83%A8%E7%BD%B2) 获取大 EP 初始化脚本，部署目录
结构如下：

```shell
boot_helper/boot.sh            # Pod 执行脚本
collect_pd_cluster_logs.sh     # 收集日志
conf/
    |_ mindie_env.json         # A2 环境变量配置
    |_ mindie_env_a3.json      # A3 环境变量配置
delete.sh                      # 删除 Pod，停止服务
deploy_ac_job.py               # 启动部署，通常不需要修改
deployment/                    # Kubernetes 配置，通常不需要修改
user_config.json               # 针对 A2 硬件的服务配置
user_config_base_A3.json       # 针对 A3 硬件的服务配置
```

#### 配置服务化环境变量

修改 `conf/mindie_env.json` （A3 则修改 `conf/mindie_env_a3.json`）。

```json
{
  "mindie_common_env": {
    ...
    "TASK_QUEUE_ENABLE": 0,            # 关闭任务队列，避免多流场景下精度问题
    "HCCL_BUFFSIZE": 1050,
    "HCCL_EXEC_TIMEOUT": 1200,         # 增大数值避免超时，注意：旧版本 boot_helper/boot.sh 中重复定义该变量，一并增大数值
    "MASTER_PORT": 10000               # 增加此行，如果端口被占用，则把值修改为其他空闲的端口
  },
  "mindie_server_prefill_env": {
    ...
    "HCCL_OP_EXPANSION_MODE": "HOST",  # 使用 HOST 模式，避免通信算子偶发报错
    "NPU_MEMORY_FRACTION": 0.8
  },
  "mindie_server_decode_env": {
    ...
    "TASK_QUEUE_ENABLE": 0,            # 关闭任务队列，避免多流场景下精度问题
    "NPU_MEMORY_FRACTION": 0.8,
    "HCCL_CONNECT_TIMEOUT": 7200,
    "HCCL_OP_EXPANSION_MODE": "HOST",  # 使用 HOST 模式，避免通信算子偶发报错
    ...
  }
}
```

#### 配置服务化参数

修改 `user_config.json`（A3 则修改 `user_config_base_A3.json`）。

```json
{
  "deploy_config": {
    "p_instances_num": 1,
    "d_instances_num": 1,
    "single_p_instance_pod_num": 4,
    "single_d_instance_pod_num": 4,
    "p_pod_npu_num": 8,
    "d_pod_npu_num": 8,
    "image_name": "mindie:xxx",     # 设置使用的镜像
    ...
  },
  "mindie_server_prefill_config": {
    ...
    "BackendConfig": {
      "npuDeviceIds": [
        [
          0,
          1,
          2,
          3,
          4,
          5,
          6,
          7
        ]
      ],
      "tokenizerProcessNumber": 1,
      "multiNodesInferEnabled": true,
      ...
      "ModelDeployConfig": {
        ...
        "ModelConfig": [
          {
            "modelInstanceType": "Standard",
            "modelName": "DeepSeek-V3.2",
            "modelWeightPath": "/path/to/DeepSeek-V3.2",               // 权重路径，根据实际情况修改
            "worldSize": 8,
            ...
            "backendType": "torch",
            ...
            "dp": 1,                                                   // 下面几行配置dp、cp和dp等，需要手动增加
            "cp": 32,
            "tp": 1,
            "sp": 1,
            "moe_ep": 32,
            "pp": 1,
            "moe_tp": 1,
            "plugin_params": "{\"plugin_type\":\"mtp\",\"num_speculative_tokens\": 2}",  // 使能 mtp，每次生成2个token
            ...
          }
        ]
      },
      ...
    }
  },
  "mindie_server_decode_config": {
    ...
    "BackendConfig": {
      ...
      "ModelDeployConfig": {
        ...
        "ModelConfig": [
          {
            ...
            "modelName": "DeepSeek-V3.2",
            "modelWeightPath": "/path/to/DeepSeek-V3.2",
            ...
            "backendType": "torch",
            "dp": 16,  # 手动增加dp等配置
            "cp": 1,
            "tp": 2,
            "sp": 1,
            "moe_ep": 32,
            "pp": 1,
            "moe_tp": 1,
            "plugin_params": "{\"plugin_type\":\"mtp\",\"num_speculative_tokens\": 2}"
          }
        ]
      },
      ...
    }
  }
}
```

#### 拉起服务

在 Kubernetes 环境中执行：

```shell
# cd 到部署目录

# A2 环境
python3 deploy_ac_job.py
# 或
python3 deploy_ac_job.py --user_config_path user_config.json

# A3 环境
python3 deploy_ac_job.py --user_config_path user_config_base_A3.json
```

记录日志：

```shell
bash collect_pd_cluster_logs.sh
```

服务启动大约需要 **10 分钟**。查看服务状态：

**方法一**：通过日志查看

```shell
bash collect_pd_cluster_logs.sh
# 当 mindie-coordinator 的日志中出现 "MindIE-MS coordinator is ready!" 时，则认为服务启动成功
```

**方法二**：通过 Pod 状态查看

```shell
kubectl get pod -n mindie
# 当 mindie-coordinator-master-0 的 READY 状态为 1/1，则说明服务已经启动成功
```

输出示例如下：

```shell
bash-5.1# kubectl get pod -n mindie
NAME                          READY   STATUS    RESTARTS   AGE
mindie-controller-master-0    1/1     Running   0          23m
mindie-coordinator-master-0   1/1     Running   0          23m
mindie-server-d0-master-0     1/1     Running   0          23m
mindie-server-d0-worker-0     1/1     Running   0          23m
mindie-server-d0-worker-1     1/1     Running   0          23m
mindie-server-d0-worker-2     1/1     Running   0          23m
mindie-server-p0-master-0     1/1     Running   0          23m
mindie-server-p0-worker-0     1/1     Running   0          23m
mindie-server-p0-worker-1     1/1     Running   0          23m
mindie-server-p0-worker-2     1/1     Running   0          23m
```

#### 功能验证

单条 curl 验证：

```shell
curl http://127.0.0.1:31015/v1/chat/completions -X POST -d '{
    "model": "DeepSeek-V3.2",
    "messages": [{ "role": "user", "content": "What is deep learning?" }],
    "stream": false,
    "temperature": 1.0,
    "max_tokens": 100
}'
```

其中 IP 和端口（默认为 31015）根据实际情况修改，model 和配置中指定的 modelName 保持一致。

返回示例：

```text
{"id":"endpoint_common_369","object":"chat.completion","created":1774838070,"model":"DeepSeek-V3.2","choices":[{"index":0,"message":{"role":"assistant","content":"Of course! Here is a comprehensive explanation of deep learning, broken down for clarity.\n\n### The Short Answer (The Elevator Pitch)\n\n**Deep learning is a subfield of machine learning that uses artificial neural networks with many layers (\"deep\" networks) to learn and make intelligent decisions from vast amounts of data.**\n\nThink of it as a way to automatically find complex patterns in data (like images, sound, or text) by passing it through a multi-layered processing system, where each layer extracts a","tool_calls":[]},"logprobs":null,"finish_reason":"length"}],"usage":{"prompt_tokens":9,"prompt_tokens_details":{"cached_tokens":0},"completion_tokens":100,"completion_tokens_details":{"reasoning_tokens":0},"total_tokens":109}}
```

#### 停止服务

```shell
bash delete.sh
```

#### 128K 上下文配置

对于 128K 上下文，P节点使能 chunked prefill，A2 4+4 8机大EP配置如下：

```json
{
  "deploy_config": {
    "p_instances_num": 1,
    "d_instances_num": 1,
    "single_p_instance_pod_num": 4,
    "single_d_instance_pod_num": 4,
    "p_pod_npu_num": 8,
    "d_pod_npu_num": 8,
    ...
  "mindie_server_prefill_config": {
    ...
    "BackendConfig": {
      ...
      "ModelDeployConfig": {
        ...
        "ModelConfig": [
          {
            ...
            "dp": 4,
            "cp": 1,
            "tp": 8,
            "sp": 1,
            "moe_ep": 32,
            "pp": 1,
            "plugin_params": "{\"plugin_type\":\"mtp, splitfuse\",\"num_speculative_tokens\": 2}",   # 增加 splitfuse 特性
            "moe_tp": 1,
            ...
          }
        ]
      },
      "ScheduleConfig": {
        "templateType": "Mix",          # 如果无此配置，手动增加该行
        ...
        "maxPrefillBatchSize": 10,
        "maxPrefillTokens": 8192        # 每个chunk的token数量
      }
    }
  },
  "mindie_server_decode_config": {
    ...
    "BackendConfig": {
      ...
      "ModelDeployConfig": {
        "maxSeqLen": 128000,
        "maxInputTokenLen": 128000,
        "truncation": 0,
        "ModelConfig": [
          {
            ...
            "dp": 4,
            "cp": 1,
            "tp": 8,
            "sp": 1,
            "moe_ep": 32,
            "pp": 1,
            "moe_tp": 1,
            "plugin_params": "{\"plugin_type\":\"mtp\",\"num_speculative_tokens\": 2}",
            ...
          }
        ]
      },
      "ScheduleConfig": {
        ...
        "maxPrefillTokens": 128000,    # 校验需要，decode中不会读取该值
        "maxBatchSize": 64,
        "maxIterTimes": 128000,
        ...
      }
    }
  }
}
```

## 精度测试

首先以 gsm8k 数据集为例，介绍精度测试方法。

**1、开源数据集获取**

首先以 gsm8k 数据测试为例，说明如何使用 aisbench。

请参考[数据集准备指南](https://github.com/AISBench/benchmark/blob/master/docs/source_zh_cn/get_started/datasets.md)获取开源数据集，找到 gsm8k 数据集，下载解压后放置于 AISBench 工具根路径的 datasets 文件夹下。

```shell
# 进入 aisbench 安装目录，以默认安装路径为例 "/usr/local/lib/python3.11/site-packages/ais_bench"
cd /usr/local/lib/python3.11/site-packages/ais_bench
# 进入 datasets 目录
cd datasets
# 将已下载 gsm8k 数据集压缩包（path/to/gsm8k.zip需要改为实际路径）拷贝到当前目录，并解压
cp path/to/gsm8k.zip ./
unzip gsm8k.zip
```

**2、修改配置**

aisbench 已经有多种配置模板，本测试主要用到 `<aisbench_install_path>/benchmark/configs/models/vllm_api/` 下面的文件，主要包括：

- `vllm_api_general_chat.py`：非流式推理配置，精度可以使用非流失，也可以使用流式
- `vllm_api_stream_chat`：流式推理配置，测试性能需要使用流式
- `vllm_api_function_call_chat.py`：测试BFCL数据集时会用到function call

以修改 `vllm_api_general_chat.py` 为例，配置内容示例如下：**

```shell
# 进入 aisbench 安装目录，不同的环境该路径可能有区别
cd /usr/local/lib/python3.11/site-packages/ais_bench
# 编辑文件
vim benchmark/configs/models/vllm_api/vllm_api_general_chat.py
```

`vllm_api_general_chat.py`内容示例：

```python
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="/path/to/DeepSeek-V3.2",  # 指定模型序列化词表文件绝对路径，一般来说就是模型权重文件夹路径
        model="DeepSeek-V3.2",          # 指定服务端已加载模型名称，和配置中的 "modelName" 值一致
        request_rate=0,
        retry=2,
        host_ip="127.0.0.1",            # 指定推理服务的IP（如果是多机推理场景则为主节点IP）
        host_port=1025,                 # 指定推理服务的端口（推理服务化config.json文件中配置的port）
        max_out_len=4096,               # 推理服务输出的token的最大数量
        batch_size=32,                  # 请求发送的最大并发数，根据服务端负载调整
        trust_remote_code=False,
        generation_kwargs=dict(
            # 需要设置特定参数时，取消对应的注释
            #top_p=0.95,
            #top_k=20,
            #seed=None,
            #temperature=1.0,
            #chat_template_kwargs={"enable_thinking": True},  # 是否 enable thinking
        )
    )
]
```

**3、执行以下命令启动服务化精度测试。**

```shell
ais_bench --models vllm_api_general_chat --datasets gsm8k_gen_4_shot_cot_chat_prompt --debug
```

其中 `--models` 的值即为所用的配置文件名。

回显如下所示则表示执行成功：

```shell
| dataset | version | metric | mode | vllm-api-stream-chat |
|----- | ----- | ----- | ----- | -----|
| gsm8k | e3c4be | accuracy | gen | 94.69 |
```

如需了解更多 AISBench 工具的使用方法，请参考：[AISBench_benchmark](https://github.com/AISBench/benchmark) 项目。

汇总各数据集精度测试配置如下：

| 数据集名称   | 配置参数                   | aisbench请求命令    | 精度结果         |
| ---          | ---                       | ---                | ---              |
| gsm8k        | max_out_len:4096          |  ais_bench --models vllm_api_general_chat --datasets gsm8k_gen_4_shot_cot_chat_prompt --debug                        | 94.69(94.69%) |
| ceval        | max_out_len:32000<br/>enable_thinking:True     |  ais_bench --models vllm_api_general_chat --datasets  ceval_gen_0_shot_cot_chat_prompt --mode all                    | 92.1(92.1%)   |
| gpqa-diamond | max_out_len:32000<br/>enable_thinking:True     | ais_bench --models vllm_api_general_chat --datasets  gpqa_gen_0_shot_cot_chat_prompt --mode all | 84.85(84.85%) |
| aime2024     | max_out_len:32000<br/>enable_thinking:True     | ais_bench --models vllm_api_general_chat --datasets  aime2024_gen_0_shot_chat_prompt --mode all | 90.0(90.0%)   |
| bfcl-simple  | max_out_len:32000<br/>temperature:0.001<br/>enable_thinking:True     | ais_bench --models vllm_api_function_call_chat --datasets BFCL_gen_simple | 0.93(93%)     |

- 说明：精度结果会存在波动，建议多次测量。

## 性能测试

**1、开源数据集获取**

测试性能时，可以根据测试的输入输出长度以及数据条数，构造 gsm8k 格式的数据，然后将生成的数据集替换 `datasets/gsm8k/test.jsonl`。

**2、配置 `vllm_api_stream_chat.py` 文件，配置内容示例如下：**

```shell
# 进入 aisbench 安装目录，以默认安装路径为例 "/usr/local/lib/python3.11/site-packages/ais_bench"
cd /usr/local/lib/python3.11/site-packages/ais_bench
# 编辑 vllm_api_stream_chat.py 文件，内容可参考如下 Python 代码示例
vim benchmark/configs/models/vllm_api/vllm_api_stream_chat.py
```

```python
from ais_bench.benchmark.models import VLLMCustomAPIChatStream

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="",                    # 指定模型序列化词表文件绝对路径，一般来说就是模型权重文件夹路径
        model="DeepSeek-V3.2",      # 指定服务端已加载模型名称，依据实际推理服务拉取的模型名称配置（配置成空字符串会自动获取）
        request_rate=0,             # 请求发送频率，每1/request_rate秒发送1个请求给服务端，小于0.1则一次性发送所有请求
        retry=2,
        host_ip="127.0.0.1",        # 指定推理服务的IP（如果是多机推理场景则为主节点IP）
        host_port=1025,             # 指定推理服务的端口（推理服务化config.json文件中配置的port）
        max_out_len=1024,           # 推理服务输出的token的最大数量
        batch_size=16,              # 请求发送的最大并发数
        generation_kwargs=dict(
            temperature=0,          # 为0时将关闭后处理，可以根据需要设置为其他值
            ignore_eos=True,        # 测试定长输出时，需要将 ignore_eos 设置为 True
        )
    )
]
```

**3、执行以下命令启动服务化性能测试。**

```shell
ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_0_shot_cot_str_perf --mode perf --summarizer default_perf --debug
```

回显如下所示则表示性能测试成功（仅示例）：

```shell
╒═══════════════════════╤═══════╤══════════╤══════╤══════╤══════╤══════╤══════╤══════╤═══╕
│ Performance Parameters│ Stage │ Average  │ Min  │ Max  │Median│  P75 │  P90 │  P99 │ N │
╞═══════════════════════╪═══════╪══════════╪══════╪══════╪══════╪══════╪══════╪══════╪═══╡
│ E2EL                  │ total │ xxx ms   │ xxx  │ xxx  │ xxx  │ xxx  │ xxx  │ xxx  │ x │
├───────────────────────┼───────┼──────────┼──────┼──────┼──────┼──────┼──────┼──────┼───┤
│ TTFT                  │ total │ xxx ms   │ xxx  │ xxx  │ xxx  │ xxx  │ xxx  │ xxx  │ x │
├───────────────────────┼───────┼──────────┼──────┼──────┼──────┼──────┼──────┼──────┼───┤
│ TPOT                  │ total │ xxx ms   │ xxx  │ xxx  │ xxx  │ xxx  │ xxx  │ xxx  │ x │
├───────────────────────┼───────┼──────────┼──────┼──────┼──────┼──────┼──────┼──────┼───┤
│ ITL                   │ total │ xxx ms   │ xxx  │ xxx  │ xxx  │ xxx  │ xxx  │ xxx  │ x │
├───────────────────────┼───────┼──────────┼──────┼──────┼──────┼──────┼──────┼──────┼───┤
│ InputTokens           │ total │ xxxx     │ xxxx │ xxxx │ xxxx │ xxxx │ xxxx │ xxxx │ x │
├───────────────────────┼───────┼──────────┼──────┼──────┼──────┼──────┼──────┼──────┼───┤
│ OutputTokens          │ total │ xxxx     │ xxxx │ xxxx │ xxxx │ xxxx │ xxxx │ xxxx │ x │
├───────────────────────┼───────┼──────────┼──────┼──────┼──────┼──────┼──────┼──────┼───┤
│ OutputTokenThroughput │ total │ xxx tok/s│ xxx  │ xxx  │ xxx  │ xxx  │ xxx  │ xxx  │ x │
╘═══════════════════════╧═══════╧══════════╧══════╧══════╧══════╧══════╧══════╧══════╧═══╛
╒════════════════════════════╤═══════╤══════════╕
│ Common Metric              │ Stage │ Value    │
╞════════════════════════════╪═══════╪══════════╡
│ Benchmark Duration         │ total │ xxx ms   │
├────────────────────────────┼───────┼──────────┤
│ Total Requests             │ total │ xxx      │
├────────────────────────────┼───────┼──────────┤
│ Failed Requests            │ total │ xxx      │
├────────────────────────────┼───────┼──────────┤
│ Success Requests           │ total │ xxx      │
├────────────────────────────┼───────┼──────────┤
│ Concurrency                │ total │ xxx      │
├────────────────────────────┼───────┼──────────┤
│ Max Concurrency            │ total │ xxx      │
├────────────────────────────┼───────┼──────────┤
│ Request Throughput         │ total │ xxx req/s│
├────────────────────────────┼───────┼──────────┤
│ Total Input Tokens         │ total │ xxx      │
├────────────────────────────┼───────┼──────────┤
│ Prefill Token Throughput   │ total │ xxx tok/s│
├────────────────────────────┼───────┼──────────┤
│ Total generated tokens     │ total │ xxx      │
├────────────────────────────┼───────┼──────────┤
│ Input Token Throughput     │ total │ xxx tok/s│
├────────────────────────────┼───────┼──────────┤
│ Output Token Throughput    │ total │ xxx tok/s│
├────────────────────────────┼───────┼──────────┤
│ Total Token Throughput     │ total │ xxx tok/s│
╘════════════════════════════╧═══════╧══════════╛
```

**关键性能指标说明：**

- **TTFT**（Time To First Token）：首 token 延迟，从发送请求到收到第一个输出 token 的时间，反映 prefill 阶段的速度。
- **TPOT**（Time Per Output Token）：每 token 延迟，decode 阶段平均生成一个 token 所需的时间，反映推理吞吐能力。
- **Prefill Token Throughput**：预填充吞吐量，prefill 阶段每秒处理的 token 数，衡量长上下文场景下首字响应的效率。
- **Output Token Throughput**：输出吞吐量，即 decode 吞吐，每秒生成的输出 token 数，反映推理阶段的实际生成能力。

---

## 典型配置

DeepSeek V3.2 的典型部署方式如下：

| 部署形态 |部署方式 |  机器数量 | 卡数 | 最大上下文 | 并行策略 | mtp 量化<br/>mtp=2 | chunked prefill | HCCL_BUFFSIZE（MB） |  NPU_MEM_FRACTION |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| A2 四机   | PD 混部 | 4 | 32  | 32K | MLA: DP4+TP8<br/>MOE: EP32+TP1 | ✅ | ❌  | 1050 | 0.92 |
| A2 四机   | PD 混部 | 4 | 32  | 64K | MLA: DP4+TP8<br/>MOE: EP32+TP1 | ❌ | ✅  | 1050 | 0.92 |
| A2 大 EP  | PD 分离 | 8 | 64  | 64K | P:<br/>MLA: DP1+TP2+CP16<br/>MOE: EP32+TP1<br/>D:<br/>MLA: DP8+TP4<br/>MOE: EP32+TP1 | ✅ | ❌ | 1050 | 0.8 |
| A2 大 EP | PD 分离 | 8 | 64  | 128K | P:<br/>MLA: DP4+TP8<br/>MOE: EP32+TP1<br/>D:<br/>MLA: DP4+TP8<br/>MOE: EP32+TP1 | ✅ | P✅ D❌ | 1050 | 0.8 |
| A3 双机  | PD 混部 | 2 | 16  | 32K | MLA: DP4+TP8<br/>MOE: EP32+TP1 | ✅ | ❌ | 1050 | 0.92 |
| A3 双机  | PD 混部 | 2 | 16  | 64K | MLA: DP4+TP8<br/>MOE: EP32+TP1 | ❌ | ✅ | 1050 | 0.92 |
| A3 大 EP | PD 分离 | 4 | 32  | 64K | P:<br/>MLA: DP1+TP2+CP16<br/>MOE: EP32+TP1<br/>D:<br/>MLA: DP8+TP4<br/>MOE: EP32+TP1 | ✅ | ❌ | 1050 | 0.8 |
| A3 大 EP | PD 分离 | 4 | 32  | 128K | P:<br/>MLA: DP4+TP8<br/>MOE: EP32+TP1<br/>D:<br/>MLA: DP4+TP8<br/>MOE: EP32+TP1 | ✅ | P✅ D❌ | 1050 | 0.8 |

---

## 声明

- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集来完成示例，请您特别注意应遵守对应数据集和模型的 License，如您因使用数据集或者模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本地代码的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交 issue，我们将及时审视并解答。
