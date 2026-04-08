# W4A8混合量化

## 简介

混合量化是对模型的不同层级采用不同的量化方式。DeepSeek R1/V3的W4A8混合量化：前三层MLP是W8A8 Dynamic量化，MLA&共享专家层是W8A8量化，路由专家层是W4A8 Dynamic量化。其中W4A8 Dynamic量化采用Per-channel和Per-group对权重进行4bit量化，对激活进行8bit量化。

> [!NOTE]说明 
>-  仅支持DeepSeek-R1，DeepSeek-V3模型。
>-  仅支持Anti-Outlier离群值处理、暂不支持KV Cache int8量化配合使用。.
>-  如需开启共享专家混置特性，需要更换为共享专家层量化为W4A8的特殊权重，且保持开启。

量化后权重目录结构：

```
├─ config.json
├─ quant_model_weight_w8a8.safetensors
├─ quant_model_description.json
├─ tokenizer_config.json
├─ tokenizer.json
└─ tokenizer.model
```

-   量化输出包含：权重文件quant\_model\_weight\_w8a8.safetensors和权重描述文件quant\_model\_description.json。
-   目录中的其余文件为推理时所需的配置文件，不同模型略有差异。

以下展示了量化后权重描述文件quant\_model\_description.json中的部分内容：

```
{
  "model_quant_type": "W8A8_DYNAMIC",
  "model.embed_tokens.weight": "FLOAT",
  "model.layers.0.self_attn.q_proj.weight": "W8A8",
  "model.layers.0.self_attn.q_proj.weight_scale": "W8A8",
  "model.layers.0.self_attn.q_proj.weight_offset": "W8A8",
   ...
   "model.layers.1.mlp.gate_proj.weight": "W8A8_DYNAMIC",
   "model.layers.1.mlp.gate_proj.weight_scale": "W8A8_DYNAMIC",
   "model.layers.1.mlp.gate_proj.weight_offset": "W8A8_DYNAMIC",
   ...
  "model.layers.3.mlp.experts.0.gate_proj.weight": "W4A8_DYNAMIC",
  "model.layers.3.mlp.experts.0.gate_proj.weight_scale": "W4A8_DYNAMIC",
  "model.layers.3.mlp.experts.0.gate_proj.weight_scale_second": "W4A8_DYNAMIC",
   "model.layers.3.mlp.experts.0.gate_proj.scale_bias": "W4A8_DYNAMIC",
  ...
}
```

量化后的MatMul权重新增weight\_scale、weight\_scale\_second和scale\_bias，用于对MatMul的计算结果进行反量化。

**图 1**  量化权重推理时流程<a name="fig132131518185315"></a>  
![](./figures/w4a8_mixed_precision_quantization.png "量化权重推理时流程-0")

此量化方式支持量化float16或bfloat16类型的原始权重。

**表 1**  float16权重量化后dtype及shape信息（假设原始权重的shape为\[n, k\]）

|Tensor信息|weight|weight_scale|weight_scale_second|scale_bias|
|--|--|--|--|--|
|dtype|int4|float32|float32|uint64|
|shape|[n, k]|[n, 1]|[n, group_num]|[n,group_num]|


**表 2**  bfloat16权重量化后dtype及shape信息（假设原始权重的shape为\[n, k\]）

|Tensor信息|weight|weight_scale|weight_scale_second|scale_bias|
|--|--|--|--|--|
|dtype|int4|bfloat32|bfloat32|uint64|
|shape|[n, k]|[n, 1]|[n, group_num]|[n,group_num]|


> [!NOTE]说明 
> 仅当浮点权重存在bias场景时，量化权重才会有bias。

