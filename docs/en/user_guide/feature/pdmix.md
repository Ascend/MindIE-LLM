# PDMIX Quantization

## Overview

PDMIX quantization uses different quantization modes in the prefill and decode phases of model inference.

**Table 1** PDMIX quantization

|Quantization Mode|Inference Phase|Quantization Feature|Applicable Scenario|
|--|--|--|--|
|**W8A8 Dynamic (Per-token)**|Prefill|Each token is quantized using an independent input scale, dynamically adapting to the range of activation values per token.|For accuracy-critical scenarios, dynamic quantization minimizes precision loss by adapting to the wide distribution shifts in activation values during long-sequence processing or the prompt phase.|
|**W8A8 Static (Per-tensor)**|Decode|The entire tensor is quantized with a single, fixed input scale, minimizing computational overhead.|For performance-critical scenarios, static quantization maximizes inference throughput in the token-by-token generation phase, where the compute-to-memory ratio is low.|

Weight directory structure after quantization:

```text
├─ config.json
├─ configuration.json
├─ generation_config.json
├─ quant_model_description.json
├─ quant_model_weight_w8a8_mix.safetensors
├─ tokenizer.json
└─ tokenizer_config.json
```

- Quantized outputs: `quant_model_weight_w8a8_mix.safetensors` (weight file) and `quant_model_description.json` (weight description file).
- The other files in the directory are required for inference, and they vary slightly by model.

The following shows part of the content in weight description file `quant_model_description.json` after quantization:

```json
{
  "model.layers.0.self_attn.q_proj.weight": "W8A8_MIX",
  "model.layers.0.self_attn.q_proj.quant_bias": "W8A8_MIX",
  "model.layers.0.self_attn.q_proj.input_scale": "W8A8_MIX",
  "model.layers.0.self_attn.q_proj.input_offset": "W8A8_MIX",
  "model.layers.0.self_attn.q_proj.deq_scale": "W8A8_MIX",
  "model.layers.0.self_attn.q_proj.weight_scale": "W8A8_MIX",
  "model.layers.0.self_attn.q_proj.weight_offset": "W8A8_MIX"
}
```

Compared with the W8A8 weight quantization, `weight_scale` and `weight_offset` are added to dequantize the Matmul computation result.

The inference process of weight quantization is the same as that of W8A8 quantization.

This quantization mode supports quantization of the original weights of the bfloat16 type.

**Table 2** dtype and shape information after bfloat16 weight quantization (assuming that shape of the original weight is `[n, k]`)

|Tensor|weight|quant_bias|input_scale|input_offset|deq_scale|weight_scale|weight_offset|
|--|--|--|--|--|--|--|--|
|dtype|int8|int32|bf16|bf16|fp32|bf16|bf16|
|shape|[n,k]|[n]|[1]|[1]|[n]|[n,1]|[n,1]|

> [!NOTE]
> The quantized weight has bias only when the floating-point weight has bias.

## Weight Generation

You can use the [msModelSlim](https://gitcode.com/Ascend/msmodelslim/blob/26.0.0/README_EN.md) tool to generate quantized weights.

The following uses Qwen3-14B as an example. After installing msModelSlim, you can run the following command to quickly generate the W8A8PDMIX quantization weights:

```sh
msmodelslim quant --model_path {*Floating_point_weight_path*} --save_path {*W8A8PDMIX_quantized_weight_path*} --device npu --model_type Qwen3-14B --quant_type w8a8 --trust_remote_code True
```

The preceding command is a best practice of msModelSlim. For details about more quantization parameter configurations, see the msModelSlim documentation.

## Inference

The following uses Qwen3-14B-W8A8PDMIX weights as an example. You can run the following commands to perform a dialog test, with the inference content being "What's deep learning?" and a maximum output of 20 tokens.

```sh
cd ${ATB_SPEED_HOME_PATH}
torchrun --nproc_per_node 2 --master_port 12350 -m examples.run_pa --model_path {PDMIX_quantized_weight_path}
```
