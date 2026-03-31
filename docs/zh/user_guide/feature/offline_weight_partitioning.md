# 权重离线切分

MindIE在权重加载过程中，默认实现为首先完整加载safetensor格式的权重文件，然后在内存中依据并行策略执行切分处理，最终通过H2D（Host-to-Device）方式将权重传输至NPU卡上。针对如DeepSeek等大规模参数模型，为降低权重加载时间开销，可采用权重离线切分优化策略：即预先依据运行时并行策略对权重进行切分并存储于tmpfs中，以实现更高效的加载流程。

## 限制与约束

- 仅DeepSeek-R1和DeepSeek-V3模型支持此特性。
- 权重离线切分时的配置需和模型推理运行时的配置保持一致。
- 仅Atlas 800I A2 推理服务器双机、Atlas 800I A3 超节点服务器单机场景支持此特性。
- 不支持与共享专家和路由专家合并特性同时开启。
- 不支持和动态负载均衡特性同时开启。

## 权重离线切分

以Atlas 800I A3 超节点服务器单机为例，您可以使用以下脚本完成权重切分。

```bash
# 如在线服务化运行场景使能MTP权重，请设置以下环境变量
export DEEPSEEK_MTP=1
# 权重切分
torchrun --nproc_per_node 16 --master_port 20030 -m examples.convert.weight_sharder --model_path {完整权重路径} --dp 2 --tp 8 --moe_tp 4 --moe_ep 4 --save_directory {切分后权重文件保存路径}
```

以Atlas 800I A2 推理服务器双机为例，您可以使用以下脚本完成权重切分。

```bash
# 如在线服务化运行场景使能MTP权重，请设置以下环境变量
export DEEPSEEK_MTP=1
export RANK_TABLE_FILE={ranktable文件路径}
# 权重切分
torchrun --nnodes=2 --nproc_per_node 8 --node_rank=0 --master_addr="主节点IP" --master_port 20030 -m examples.convert.weight_sharder --model_path {完整权重路径} --dp 2 --tp 8 --moe_tp 4 --moe_ep 4 --save_directory {切分后权重文件保存路径}

torchrun --nnodes=2 --nproc_per_node 8 --node_rank=1 --master_addr="主节点IP" --master_port 20030 -m examples.convert.weight_sharder --model_path {完整权重路径} --dp 2 --tp 8 --moe_tp 4 --moe_ep 4 --save_directory {切分后权重文件保存路径}
```

切分后的权重目录结构：

```text
├── config.json
├── configuration.json
├── generation_config.json
├── model-000
│   └── model.safetensors
...
├── model-015
│   └── model.safetensors
├── model-attn-tp-000
│   └── model.safetensors
...
├── model-attn-tp-007
│   └── model.safetensors
├── model-dense-tp-000
│   └── model.safetensors
...
├── model-dense-tp-007
│   └── model.safetensors
├── model-moe-tp-000-ep-000
│   ├── model-00001-of-00005.safetensors
│   ├── model-00002-of-00005.safetensors
│   ├── model-00003-of-00005.safetensors
│   ├── model-00004-of-00005.safetensors
│   └── model-00005-of-00005.safetensors
...
├── model-moe-tp-003-ep-003
│   ├── model-00001-of-00005.safetensors
│   ├── model-00002-of-00005.safetensors
│   ├── model-00003-of-00005.safetensors
│   ├── model-00004-of-00005.safetensors
│   └── model-00005-of-00005.safetensors
├── model-norm
│   └── model.safetensors
├── model_sharded_metadata.json
├── quant_model_description_w8a8_dynamic.json
├── tokenizer.json
└── tokenizer_config.json
```

> [!NOTE]说明 
>
>- 切分后模型权重按照model层、norm模块、attention模块、dense模块以及moe模块分目录存储。
>- 切分后新增model\_sharded\_metadata.json文件，用于索引切分策略和切分文件。

## 执行推理

以在线服务化推理场景为例。

1. 打开Server的config.json文件。

    ```bash
    cd {MindIE安装目录}/latest/mindie-service/
    vi conf/config.json
    ```

2. 配置服务化参数。将模型权重路径修改为切分后的权重文件保存路径，服务化参数说明请参见[配置参数说明（服务化）](../user_manual/service_parameter_configuration.md)章节，参数配置示例如下。

    ```json
    "ModelDeployConfig" :
    {
       "maxSeqLen" : 2560,
       "maxInputTokenLen" : 2048,
       "truncation" : 0,
       "ModelConfig" : [
         {
             "modelInstanceType" : "Standard",
             "modelName" : "DeepSeek-R1_w8a8",
             "modelWeightPath" : "切分后权重文件保存路径",
             "worldSize" : 8,
             "cpuMemSize" : 5,
             "npuMemSize" : -1,
             "backendType" : "atb",
             "trustRemoteCode" : false,
          }
       ]
    },
    ```
