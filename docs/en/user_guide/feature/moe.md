# MoE

There are two key innovations in the traditional transformer structure for MoE. First, it replaces the Feed-Forward Network (FFN) with a Sparse MoE layer. Each FFN acts as an expert, but only a subset is activated per token during inference. The second innovation, the routing mechanism, is crucial for selecting the subset of experts to activate. The router determines which expert the token will enter at each layer. Thanks to the two mechanisms, MoE models can ensure an excellent model effect due to extensive expert knowledge. Compared with traditional models with the same number of parameters, MoE models guarantee high-performance inference by activating only some experts.

Typical models of the MoE structure include Mixtral 8x7B, Mixtral 8x22B, DeepSeek-16B-MoE, DeepSeek-V2, DeepSeek-V3, DeepSeek-R1, Qwen3-30B-A3B, and Qwen3-235B-A22B.

## Constraints

For details about the supported feature capabilities, see [Table 1](#table1).

**Table 1** Feature support matrix <a id="table1"></a>

|Supported Models|Data format|Quantization|Parallelism|Hardware Platform|Multi-Server Multi-Card Inference|
|--|--|--|--|--|--|
|Mixtral 8x7B|FP16|Not Supported|TP|Atlas 800I A2 inference server|Not supported|
|Mixtral 8x22B|FP16|Not Supported|TP|Atlas 800I A2 inference server|Not supported|
|DeepSeek-16B-MoE|FP16|Not Supported|TP|Atlas 800I A2 inference server|Not supported|
|DeepSeek-V2|BF16|Supported|TP and EP|Atlas 800I A2 inference server|Supported|
|DeepSeek-V3|BF16|Supported|TP and EP|Atlas 800I A2 inference server|Supported|
|DeepSeek-R1|BF16|Supported|TP and EP|Atlas 800I A2 inference server|Supported|
|Qwen3-30B-A3B|BF16|Supported|TP|Atlas 800I A2 inference server|Not supported|
|Qwen3-235B-A22B|BF16|Supported|TP|Atlas 800I A2 inference server|Supported|

**Model configuration parameters**

For details about how to configure the inherent parameters of each model, see the `config.json` file in their official weight file.

## Inference

The inference method for MoE models is identical to that of other models. You can adopt the traditional LLM method during inference without setting any additional parameters.

The following uses DeepSeek-16B-MoE as an example. You can run the following commands to perform a dialog test. The inference content is "What's deep learning".

```bash
cd ${ATB_SPEED_HOME_PATH}
bash examples/models/deepseek/run_pa_deepseek_moe.sh {Model_weight_path}
```
