# Configuration Parameters (Model Side)

The path of the configuration file in the `atb-models` installation directory on the model side is `${ATB_SPEED_HOME_PATH}/atb_llm/conf/config.json`.

The format of the model configuration file `config.json` is as follows:

```json
{
  "llm": {
    "ccl": {
      "enable_mc2": "true"
    },
    "stream_options": {
      "micro_batch": "false"
    },
    "engine": {
      "graph": "cpp"
    },
    "parallel_options": {
      "o_proj_local_tp": -1,
      "dense_mlp_local_tp": -1,
      "lm_head_local_tp": -1,
      "hccl_buffer": 128,
      "hccl_moe_ep_buffer": 512,
      "hccl_moe_tp_buffer": 64
    },
    "pmcc_obfuscation_options": {
      "enable_model_obfuscation": false,
      "data_obfuscation_ca_dir": "",
      "kms_agent_port": 1024
    },
    "kv_cache_options": {
      "enable_nz": false
    },
    "weights_options": {
      "low_cpu_memory_mode": false
    },
    "enable_reasoning": "false",
    "tool_call_options": {
        "tool_call_parser": ""
    },
    "chat_template": "",
    "ep_level": 1,
    "communication_backend": {
        "prefill": "lccl",
        "decode": "lccl"
    }
  },
  "models": {
    "qwen_moe": {
      "eplb": {
        "level": 0,
        "expert_map_file": ""
      },
      "ep_level": 2
    },
    "deepseekv2": {
      "eplb": {
        "level": 0,
        "expert_map_file": "",
        "num_redundant_experts": 0,
        "aggregate_threshold": 128,
        "num_expert_update_ready_countdown": 16
      },
      "ep_level": 1,
      "enable_dispatch_combine_v2": true,
      "communication_backend": {
        "prefill": "lccl",
        "decode": "lccl"
      },
      "mix_shared_routing": false,
      "enable_gmmswigluquant": false,
      "enable_oproj_prefetch": false,
      "enable_mlapo_prefetch": false,
      "num_dangling_shared_experts": 0,
      "enable_swiglu_quant_for_shared_experts": false,
      "enable_init_routing_cutoff": false,
      "topk_scaling_factor": 1.0,
      "h3p":{
        "enable_qkvdown_dp": "true",
        "enable_gating_dp": "true",
        "enable_shared_expert_dp": "false",
        "enable_shared_expert_overlap": "false"
      }
    }
  }
}
```

## Parameters in `llm`

|Parameter|Value Type|Value Range|Configuration Description|
|--|--|--|--|
|enable_reasoning|bool|<ul><li>true</li><li>false</li></ul>|Whether to enable model output parsing. The output is parsed into the `reasoning content` and `content` fields. <ul><li>`false`: disabled</li><li>`true`: enabled</li></ul>This parameter is required. The default value is `false`.<br>This function can be enabled only for the Qwen3-32B, Qwen3-30B-A3B, DeepSeek-R1-671B, and DeepSeek-V3.1 models.|
|chat_template|string|<ul><li>Path to the .jinja file</li><li>""</li></ul>|Imports a custom dialog template to replace the default dialog template. <ul><li>Default value: `""`</li><li>For DeepSeek models, the default `chat_template` in `tokenizer_config.json` cannot be called using tools. You can use this parameter to input the `chat_template` that can be called using tools. </li><li>This parameter can be used to input a custom template for DeepSeek, Qwen (LLM), ChatGLM, and Llama models.</li></ul>|
|**tool_call_options**|-|-|-|
|tool_call_parser|string|<ul><li>Optional registered name in the registered ToolsCallProcessor name. For details, see [Table 2 in the Function Call](../feature/function_call.md#table2)</li><li>`""`</li></ul>|Parsing mode of the tool when the function call feature is enabled. <ul><li>Default value: `""`</li><li> If this parameter is not set or is set to an incorrect value, the default tool parsing mode corresponding to the current model is used. </li><li>When DeepSeek-V3.1 uses the function call feature, this parameter must be set to `deepseek_v31`. For other models, use the default value. </li><li>This parameter is used together with `chat_template`. The corresponding `ToolsCallProcessor` is selected based on the function call format specified in `chat_template`.</li></ul>|
|**ccl**|-|-|-|
|enable_mc2|bool|<ul><li>true</li><li>false</li></ul>|Whether to enable the communication-computing fused operator feature. <ul><li>Default value: `true`</li><li>This feature cannot be enabled together with the communication-computation dual-stream overlapping feature.</li></ul>|
|**stream_options**|-|-|-|
|micro_batch|bool|<ul><li>true</li><li>false</li></ul>|Whether to enable the communication-computing dual-stream overlapping feature. <ul><li>This feature cannot be enabled together with the communication-computing fused operator feature. </li><li>This feature cannot be enabled together with the Python graph. </li><li>Only the Qwen2.5-14B, Qwen3-14B, Deepseek-R1, and DeepSeek-V3.1 models support this feature. </li><li>Enabling this feature will cause extra graphics memory usage. In serving scenarios, if the number of KV caches decreases, scheduling will be affected and the throughput will decrease. Therefore, you are advised not to enable this feature when the graphics memory is limited. </li><li>Default value: `false`</li></ul>|
|**engine**|-|-|-|
|graph|string|<ul><li>cpp</li><li>python</li></ul>|Enables the cpp graph or Python graph. <ul><li>Only the Llama3.1-8B, Qwen2.5-7B, Qwen3-14B, and Qwen3-32B models support the Python graph. </li><li>Default value: `cpp`</li></ul>|
|**parallel_options**|-|-|-|
|o_proj_local_tp|int|[1, worldSize/Number of nodes]|Split count for the Attention O matrix. <ul><li>Only the DeepSeek-R1, DeepSeek-V3, and DeepSeek-V3.1 models support this feature. </li><li>Default value: `-1`, indicating that splitting is disabled. </li></ul>|
|lm_head_local_tp|int|[1, worldSize/Number of nodes]|TP split count for LmHead. <ul><li>Only the DeepSeek-R1, DeepSeek-V3, and DeepSeek-V3.1 models support this feature. </li><li>Default value: `-1`, indicating that splitting is disabled. </li></ul>|
|hccl_buffer|int|≥1|Buffer size of the shared data in communicators except the MoE communicator. <ul><li>Default value: `128`</li><li>If the value is too large, error message "out of memory" will be displayed. The default value is recommended.</li></ul>|
|hccl_moe_ep_buffer|int|≥512|Buffer size of the shared data in the MoE EP communicator. <ul><li>Default value: `512`</li><li>If the value is too large, error message "out of memory" will be displayed. The default value is recommended.</li></ul>|
|hccl_moe_tp_buffer|int|≥64|Buffer size of the shared data in the MoE TP communicator. <ul><li>Default value: `64`</li><li>If the value is too large, error message "out of memory" will be displayed. The default value is recommended.</li></ul>|
|**kv_cache_options**|-|-|-|
|enable_nz|bool|<ul><li>true</li><li>false</li></ul>|Whether to enable the NZ format for the KV cache. <ul><li>Only the DeepSeek-R1, DeepSeek-V3, and DeepSeek-V3.1 models support this feature. The NZ format is automatically enabled in the FA3 quantization scenario. </li><li>Default value: `false`</li></ul>|
|**weights_options**|-|-|-|
|low_cpu_memory_mode|bool|<ul><li>true</li><li>false</li></ul>|Whether to enable the low CPU and memory usage mode. <ul><li>This feature must be enabled together with the Python graph. </li><li>This feature is supported only by the Qwen2.5-7B model. </li><li> default value: `false` (disabled) </li></ul><br>After this function is enabled, model parameters will be loaded tensor by tensor in the weight loading phase, which significantly reduces the CPU and memory usage. This function is especially suitable for memory-limited scenarios such as edge devices and small-specification servers. In an environment with sufficient CPU and memory resources, you are advised to disable this function to reduce the loading time.|

## Parameters in `models`

|Parameter|Value Type|Value Range|Configuration Description|
|--|--|--|--|
|deepseekv2|map|-|DeepSeek-V2 configuration. For details, see [DeepSeek-V2 parameters](#deepseek-v2-parameters)|

## DeepSeek-V2 parameters

|Parameter|Value Type|Value Range|Configuration Description|
|--|--|--|--|
|ep_level|int|[1,2]|EP implementation form. `1`: EP based on AllGather communication<br>`2`: EP based on AllToAll and communication-computing fusion|
|topk_scaling_factor|float|(0,1]|TopK truncation parameter. <ul><li>When `ep_level` is set to `1`, the latter part of `hidden_states` of each device is invalid data. You can set the truncation parameter to reduce the graphics memory overhead. </li><li>In addition, `enable_init_routing_cutoff` must be set to `true`.</li></ul>|
|enable_init_routing_cutoff|bool|<ul><li>true</li><li>false</li></ul>|Whether to allow topK truncation. <ul><li>Default value: `false` (disabled) </li><li>This parameter can be configured when `ep_level` is set to `1`.</li></ul>|
|alltoall_ep_buffer_scale_factors|list[list[int, float]]|Each member in the list contains two numbers. The first number is a non-negative integer, and the second number is a floating-point number greater than 0.<br>The members are sorted in descending order based on the first number.|Size of the AllToAll communication buffer. The second-level list contains two elements. The first number is the sequence length, and the second number is the buffer coefficient. The sequence length is the condition for selecting the buffer coefficient. The following is an example:<br>`[[1048576, 1.32], [524288, 1.4], [262144, 1.53], [131072, 1.8], [32768, 3.0], [8192, 5.2], [0, 8.0]]`<ul><li> This parameter is recommended when `ep_level` is set to `2` and users need to manage the graphics memory in a refined manner. </li><li>This parameter does not take effect when `ep_level` is set to `1`.</li></ul>|
|num_dangling_shared_experts|int|Non-negative integer|Number of external shared experts.<br>Currently, only the Atlas 800I A3 SuperPoD server with 144 cards and without load balancing is supported. The recommended value is `32`.<br>The default value is `0` (disabling the feature).|
|enable_mlapo_prefetch|bool|<ul><li>true</li><li>false</li></ul>|Enables or disables mlapo prefetch. <ul><li>`true`: enabled</li><li>`false`: disabled</li></ul>Default value: `false`|
|enable_oproj_prefetch|bool|<ul><li>true</li><li>false</li></ul>|Enables or disables oproj prefetch.<br>You are advised not to enable this function for the Atlas 800I A2 inference server. It is recommended that this function be enabled together with `o_proj_local_tp` on the Atlas 800I A3 SuperPoD Server. The recommended value of `o_proj_local_tp` is `2`. <ul><li>`true`: enabled</li><li>`false`: disabled</li></ul>Default value: `false`|
|**eplb**|-|-|-|
|level|int|[0, 3]|<ul><li>`0`: Disable load balancing.</li><li>`1`: Enable static redundancy load balancing.</li><li>`2`: Enable dynamic redundancy load balancing (not currently supported).</li><li>`3`: Enable forced load balancing.</li></ul>Default: `0`|
|expert_map_file|string|File Path|Path of the expert deployment table for static load balancing in redundancy mode.<br>Default value: `""`|
|num_redundant_experts|int|[0, n_routed_experts]|**This parameter is not supported in the current version.**<br>Number of redundant experts.<br>Default value: `0`|
|aggregate_threshold|int|≥1|**This parameter is not supported in the current version.**<br>Frequency of triggering the dynamic EPLB algorithm, in the unit of decoding times.<br>For example, the value `50` indicates that the dynamic EPLB algorithm is triggered once every 50 decoding times. If the algorithm determines that the popularity exceeds a certain threshold, the routing table is adjusted to reduce the popularity.|
|buffer_expert_layer_num|int|[1, num_moe_layers]|**This parameter is not supported in the current version.**<br>Number of layers transferred by dynamic EPLB each time.<br>Because weight transfer is asynchronous, an additional buffer memory is required to hold the weights without disrupting the current decode process. When it is set to 1 layer, only one layer is loaded at a time, after which its weights and routing table are flushed.<br>The formula for calculating the affected memory is as follows: `buffer_expert_layer_num` × `local_experts_num` × 44MB (44MB is the size of an int8 expert).|
|num_expert_update_ready_countdown|int|≥1|**This parameter is not supported in the current version.**<br>Frequency of checking whether the host-to-device transfer is complete, in the unit of decoding times.<br>Because weight transfer is asynchronous, the weight and routing table can be updated only after all EP cards are transferred. Communication is introduced here. When there are a large number of transfer layers, the frequency can be reduced to lower the overhead on the EPLB framework side.|
|**h3p**|-|-|-|
|enable_qkvdown_dp|bool|<ul><li>true</li><li>false</li></ul>|Whether to enable the "qkvdown dp" feature to reduce the computing and communication traffic and improve the performance in the prefill phase.<br>Default value: `true`|
|enable_gating_dp|bool|<ul><li>true</li><li>false</li></ul>|Whether to enable the "gating dp" feature to reduce the computing and communication traffic and improve the performance in the prefill phase.<br>Default value: `true`<br>This feature is supported only when `ep_level` is set to `1`.|
|enable_shared_expert_dp|bool|<ul><li>true</li><li>false</li></ul>|Whether to enable the "shared expert dp" feature to improve the performance in the prefill phase.<br>Default value: `false` <ul><li>This feature is supported only when `ep_level` is set to `1`. </li><li>Enabling this feature will occupy extra graphics memory, which may cause the "out of memory" error. You are advised to use the default value.</li></ul>|
|enable_shared_expert_overlap|bool|<ul><li>true</li><li>false</li></ul>|Whether to enable the communication-computing dual-stream overlapping feature for shared experts to improve the performance in the prefill phase in specific scenarios (the input sequence length is 2k to 16k).<br>Default value: `false` <ul><li>This feature is supported only when `ep_level` is set to `1` and `enable_shared_expert_dp` is set to `true`. </li><li>Enabling this feature will occupy extra graphics memory, which may cause the "out of memory" error. You are advised to use the default value.</li></ul>|
|enable_dispatch_combine_v2|bool|<ul><li>true</li><li>false</li></ul>|When `ep_level` is set to `2`, enabling v2 of dispatch and combine operators improves Decode stage performance.<br>Default value: `true`|
|mix_shared_routing|bool|<ul><li>true</li><li>false</li></ul>|Whether to merge shared experts and route experts to achieve parallel computing for them. <ul><li>This feature cannot be used together with the CP feature. </li><li>When PD disaggregation is enabled, this function can be enabled only on node D. </li><li>Default value: `false`</li></ul>|
