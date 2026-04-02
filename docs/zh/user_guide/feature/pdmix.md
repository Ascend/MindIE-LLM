# PDMIX量化

## 简介

PDMIX量化是指在模型推理的Prefill阶段和Decode阶段使用不同的量化方式。

**表 1**  PDMIX量化

|量化方式|推理阶段|量化特点|适用场景|
|--|--|--|--|
|**W8A8 Dynamic (Per-token)**|Prefill|每个 Token 使用独立的 Input Scale 进行量化，能够动态适应不同 Token 的激活值范围。|**精度优先**。在处理长序列或 Prompt 阶段，激活值分布变化较大，动态量化能显著减少精度损失。|
|**W8A8 Static (Per-tensor)**|Decode|整个 Tensor 使用统一的 Input Scale 进行量化，参数固定，计算开销最小。|**性能优先**。在逐个生成 Token 的阶段，计算访存比（Compute-to-Memory Ratio）较低，静态量化能最大化推理吞吐率。|

量化后权重目录结构：

```
├─ config.json
├─ configuration.json
├─ generation_config.json
├─ quant_model_description.json
├─ quant_model_weight_w8a8_mix.safetensors
├─ tokenizer.json
└─ tokenizer_config.json
```

-  量化输出包含：权重文件`quant_model_weight_w8a8_mix.safetensors`和权重描述文件`quant_model_description.json`。
-  目录中的其余文件为推理时所需的配置文件，不同模型略有差异。

以下展示了量化后权重描述文件`quant_model_description.json`中的部分内容：

```json
{
  "model.layers.0.self_attn.q_proj.weight": "W8A8_MIX",
  "model.layers.0.self_attn.q_proj.quant_bias": "W8A8_MIX",
  "model.layers.0.self_attn.q_proj.input_scale": "W8A8_MIX",
  "model.layers.0.self_attn.q_proj.input_offset": "W8A8_MIX",
  "model.layers.0.self_attn.q_proj.deq_scale": "W8A8_MIX",
  "model.layers.0.self_attn.q_proj.weight_scale": "W8A8_MIX",
  "model.layers.0.self_attn.q_proj.weight_offset": "W8A8_MIX",
}
```

与W8A8量化权重相比，新增`weight_scale`和`weight_offset`，用于对Matmul的计算结果进行反量化。

量化权重推理流程同W8A8量化。

此量化方式支持量化bfloat16类型的原始权重。

**表 2**  bfloat16权重量化后dtype及shape信息（假设原始权重的shape为 `[n, k]`）

|Tensor信息|weight|quant_bias|input_scale|input_offset|deq_scale|weight_scale|weight_offset|
|--|--|--|--|--|--|--|--|
|dtype|int8|int32|bf16|bf16|fp32|bf16|bf16|
|shape|[n,k]|[n]|[1]|[1]|[n]|[n,1]|[n,1]|


> [!NOTE]说明 
> 仅当浮点权重存在bias场景时，量化权重才会有bias。

## 生成权重

您可以使用msModelSlim工具生成量化权重：[msModelSlim](https://gitcode.com/Ascend/msit/blob/master/msmodelslim/README.md)

以Qwen3-14B为例，安装msModelSlim工具后，可以使用如下命令快速生成一份W8A8PDMIX量化权重：

```sh
msmodelslim quant --model_path {浮点权重路径} --save_path {W8A8PDMIX量化权重路径} --device npu --model_type Qwen3-14B --quant_type w8a8 --trust_remote_code True
```

上述命令是msModelSlim工具的一个最佳实践，如需了解更多量化参数配置，请参考msModelSlim工具文档。

## 执行推理

以Qwen3-14B-W8A8PDMIX权重为例，您可以使用以下指令执行对话测试，推理内容为"What's deep learning?"，最长输出20个token。

```sh
cd ${ATB_SPEED_HOME_PATH}
torchrun --nproc_per_node 2 --master_port 12350 -m examples.run_pa --model_path {pdmix量化权重路径}
```
