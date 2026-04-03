# Micro Batch

Micro Batch 即在批处理过程中，将数据切分为更小粒度的多个 batch 运行。当前实现中，通过额外创建一条数据流，将一批数据分成两个 batch 在两条数据流上执行。数据流 1 在执行计算时，数据流 2 可进行通信，计算和通信耗时掩盖，使得硬件资源得以充分利用，以提高推理吞吐。

数据流间通过 **Event 机制**进行同步，计算和通信任务间都相互不冲突，防止硬件资源抢占。此特性通常应用在 **Prefill 阶段**，因为 Prefill 阶段通信类算子耗时较长，且通信类算子与计算类算子耗时占比更为均衡。在此实现下，计算和通信类算子掩盖率达 **70%+**。

## 限制与约束

- 此特性**不能与通信计算融合算子特性**同时开启。
- 此特性**不能与 Python 图组**同时开启。
- 此特性**仅支持和量化特性**同时开启。
- 仅 **Qwen2.5-14B**、**Qwen3-14B**、**Deepseek-R1** 和 **DeepSeek-V3.1** 模型支持此特性。
- 开启此特性后会带来**额外的显存占用**。
- 服务化场景下，KV Cache 数量下降会影响调度导致吞吐降低，在**显存受限的场景下，不建议开启**。

## 参数说明

开启 Micro Batch 特性，需要配置的参数如表 **Micro Batch 特性补充参数：ModelConfig 中的 models 参数** 所示。

### 表1 Micro Batch 特性补充参数：ModelConfig 中的 models 参数

| 配置项 | 取值类型 | 取值范围 | 配置说明 |
|--------|----------|----------|----------|
| `stream_options` → `micro_batch` | `bool` | `true` / `false` | 开启通信计算双流掩盖特性。<br>默认值：`false`（关闭） |

## 执行推理

1. 打开 Server 的 `config.json` 文件。

```bash
cd {MindIE安装目录}/latest/mindie-service/
vi conf/config.json
```

2. 配置服务化参数。在 Server 的 `config.json` 文件中添加 `"micro_batch"` 字段（如下加粗部分），参数字段说明请参见表 **Micro Batch 特性补充参数：ModelConfig 中的 models 参数**，服务化参数说明请参见 5.2-配置参数说明（服务化）章节，参数配置示例如下：

```json
"ModelDeployConfig": {
  "maxSeqLen": 2560,
  "maxInputTokenLen": 2048,
  "truncation": 0,
  "ModelConfig": [
    {
      "modelInstanceType": "Standard",
      "modelName": "Qwen3-14B",
      "modelWeightPath": "/data/weights/Qwen3-14B",
      "worldSize": 8,
      "cpuMemSize": 5,
      "npuMemSize": -1,
      "backendType": "atb",
      "trustRemoteCode": false,
      "models": {
        "qwen3": {
          "ccl": {
            "enable_mc2": false
          },
          "stream_options": {
            "micro_batch": true
          }
        }
      }
    }
  ]
}
```

3. 启动服务。

```bash
./bin/mindieservice_daemon
```
