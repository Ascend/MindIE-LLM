# GLM4-9B 模型推理指导 <!-- omit in toc -->

## 概述

- GLM4 是智谱 AI 和清华大学 KEG 实验室联合发布的对话预训练模型。GLM4-9B 是 [GLM4](https://github.com/THUDM/GLM-4) 系列中的开源模型，在保留了前三代模型对话流畅、部署门槛低等众多优秀特性的基础上，GLM4-9B 有更强大的基础模型、更完整的功能支持、和更全面的开源序列。
- 此代码仓中实现了一套基于 NPU 硬件的 GLM4-9B-chat 推理模型。配合加速库使用，旨在 NPU 上获得极致的推理性能。

## 特性矩阵

此矩阵罗列了 GLM4-9B 模型支持的特性。

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8 量化 | W8A16 量化 | KV cache 量化 | 稀疏量化 | MOE 量化 | MindIE Service | TGI | 长序列 |
|-------------|-------------------------|-------------------------|------|------|-----------------|-----------------|-----------|------------|--------------|--------|---------|----------------|-----|--------|
| GLM4-9B-chat | 支持 world size 1,2,4,8 | 支持 world size 1,2,4 | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |

- 此模型仓已适配的模型版本：
  - [GLM4-9B-chat](https://huggingface.co/THUDM/glm-4-9b-chat/tree/main)
- **注意**：GLM4-9B-chat 推荐使用 commit id 为 `e824789b14985787c181beaac940d103b2516cb0` 的模型仓版本。
- **注意**：transformer 需要安装固定的 4.42.4 版本。

## 使用说明

- 参考[此 README 文件](../../chatglm/v2_6b/README.md)。

### 量化权重导出

量化权重可通过 msmodelslim（昇腾模型压缩工具）实现。

#### 环境准备

请参考 [msmodelslim 安装指南](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/install_guide/install_guide.md) 安装 msModelSlim 量化工具。

#### 导出 w8a8 量化权重

通过 `${llm_path}/examples/models/chatglm/v4_9b/quant_glm4_w8a8.sh` 文件导出模型的量化权重（注意量化权重不要和浮点权重放在同一个目录下）：

```shell
export INF_NAN_MODE_ENABLE=0
bash quant_glm4_w8a8.sh -src ${浮点权重路径} -dst ${量化权重保存路径} -trust_remote_code
```

导出量化权重后应生成 `quant_model_weight_w8a8.safetensors` 和 `quant_model_description_w8a8.json` 两个文件。

注：

1.quant_glm4_w8a8.sh 文件中已配置好较优的量化策略，导出量化权重时可直接使用，也可修改为其它策略。

2.执行脚本生成量化权重时，会在生成的权重路径的 config.json 文件中添加(或修改)`quantize` 字段，值为相应量化方式，当前仅支持 `w8a8`。

3.执行完以上步骤后，执行量化模型只需要替换权重路径。

4.如果生成权重时遇到 `OpenBLAS Warning: Detect OpenMP Loop and this application may hang. Please rebuild the library with USE_OPENMP = 1 option`，可通过设置 `export OMP_NUM_THREADS=1` 来关闭多线程规避。

#### 导出 w8a8c8（kv cache 量化）权重

请参考 [msmodelslim GLM 量化示例](https://gitcode.com/Ascend/msmodelslim/tree/master/example/GLM)。

#### 导出稀疏量化权重

- 稀疏量化权重请使用以下指令生成

  校准数据集从 [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/e84444333b6d434ea7b0/) 获取，解压后，使用解压目录下的 `CEval/val/Other/civil_servant.jsonl` 作为校准数据集。

  ```shell
  cd ${llm_path}/examples/models/chatglm/v4_9b
  bash convert_quant_weights.sh ${浮点权重路径} ${W8A8S 量化权重路径} ${校准数据集路径} ${TP_Size} ${指定生成量化权重的卡号(使用单卡单芯)} ${指定生成稀疏权重的卡号(根据 TP 数来选择几卡几芯)} -trust_remote_code
  # 例如：bash convert_quant_weights.sh /home/data/glm-4-9b-chat /home/data/glm4_w8a8 /home/data/CEval/val/Other/civil_servant.jsonl 2 0 0,1 -trust_remote_code
  ```

### 运行操作说明

- 参考[此 README 文件](../../chatglm/v2_6b/README.md)

#### 开启多头自适应压缩特性

  ```shell
  # 需开启环境变量
  export ATB_LLM_RAZOR_ATTENTION_ROPE=1
  export ATB_LLM_RAZOR_ATTENTION_ENABLE=1
  ```

### 精度测试

- 参考[此 README 文件](../../../../tests/modeltest/README.md)
- 该模型不使用 chat 版本，不需设置 is_chat_model 参数

### 性能测试

- 参考[此 README 文件](../../../../tests/modeltest/README.md)

## FAQ

- `import torch_npu` 遇到 `xxx/libgomp.so.1: cannot allocate memory in static TLS block` 报错，可通过配置 `LD_PRELOAD` 解决。
  - 示例：`export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1:$LD_PRELOAD`
