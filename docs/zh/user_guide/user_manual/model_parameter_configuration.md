# 配置参数说明（模型侧）

模型侧atb-models安装目录下的配置文件获取路径为：$\{ATB\_SPEED\_HOME\_PATH\}/atb\_llm/conf/config.json

模型的配置文件config.json格式如下：

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
        "prefill":"lccl",
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

## llm参数

|配置项|取值类型|取值范围|配置说明|
|--|--|--|--|
|enable_reasoning|bool|<ul><li>true</li><li>false</li></ul>|是否开启模型输出解析，将输出分别解析为“reasoning content”和“content”两个字段。<ul><li>false：关闭</li><li>true：开启</li></ul>必填，默认值：false。<br>仅Qwen3-32B、Qwen3-30B-A3B、DeepSeek-R1-671B和DeepSeek-V3.1模型支持开启该功能。|
|chat_template|string|<ul><li>.jinja格式的文件路径</li><li>""</li></ul>|传入自定义的对话模板，替换模型默认的对话模板。<ul><li>默认值：""</li><li>DeepSeek系列模型，tokenizer_config.json中的默认chat_template不支持工具调用，可以使用该参数传入支持工具调用的chat_template。</li><li>DeepSeek系列、Qwen系列（大语言模型），ChatGLM系列，LLaMA系列模型支持使用该参数传入自定义模板。</li></ul>|
|tool_call_options|
|tool_call_parser|string|<ul><li>已注册ToolsCallProcessor名中的“可选注册名称”，具体参见[Function Call的表2](../feature/function_call.md#table2)</li><li>""</li></ul>|使能Function Call时，选择工具的解析方式。<ul><li>默认值：""</li><li>当未配置或配置错误值时，将使用当前模型所对应的默认工具解析方式。</li><li>DeepSeek V3.1模型使用Function Call时，必须配置为"deepseek_v31"，其余模型使用默认值。</li><li>与chat_template配合使用，根据chat_template中指定的Function Call调用格式选择相应的ToolsCallProcessor。</li></ul>|
|ccl|
|enable_mc2|bool|<ul><li>true</li><li>false</li></ul>|是否开启通信计算融合算子特性。<ul><li>默认值：true</li><li>此特性不能与通信计算双流掩盖特性同时开启。</li></ul>|
|stream_options|
|micro_batch|bool|<ul><li>true</li><li>false</li></ul>|开启通信计算双流掩盖特性。<ul><li>此特性不能与通信计算融合算子特性同时开启。</li><li>此特性不能与Python组图同时开启。</li><li>仅Qwen2.5-14B、Qwen3-14B、Deepseek-R1和DeepSeek-V3.1模型支持此特性。</li><li>开启此特性后会带来额外的显存占用。服务化场景下，KV Cache数量下降会影响调度导致吞吐降低，在显存受限的场景下，不建议开启。</li><li>默认值：false</li></ul>|
|engine|
|graph|string|<ul><li>cpp</li><li>python</li></ul>|开启cpp组图或python组图。<ul><li>仅LLaMA3.1-8B、Qwen2.5-7B、Qwen3-14B、Qwen3-32B模型支持Python组图。</li><li>默认值：cpp</li></ul>|
|parallel_options|
|o_proj_local_tp|int|[1，worldSize / 节点数]|表示Attention O矩阵切分数。<ul><li>仅DeepSeek-R1、DeepSeek-V3和DeepSeek-V3.1模型支持此特性。</li><li>默认值：-1，表示不开启切分</li></ul>|
|lm_head_local_tp|int|[1，worldSize / 节点数]|表示LmHead张量并行切分数。<ul><li>仅DeepSeek-R1、DeepSeek-V3和DeepSeek-V3.1模型支持此特性。</li><li>默认值：-1。表示不开启切分</li></ul>|
|hccl_buffer|int|≥1|表示除MoE通信域外，其余通信域共享数据的缓存区大小。<ul><li>默认值：128</li><li>设置过大会产生“out of memory”的错误提示，建议设置为默认值。</li></ul>|
|hccl_moe_ep_buffer|int|≥512|表示MoE专家并行相关通信域共享数据的缓存区大小。<ul><li>默认值：512</li><li>设置过大会产生“out of memory”的错误提示，建议设置为默认值。</li></ul>|
|hccl_moe_tp_buffer|int|≥64|表示MoE张量并行相关通信域共享数据的缓存区大小。<ul><li>默认值：64</li><li>设置过大会产生“out of memory”的错误提示，建议设置为默认值。</li></ul>|
|kv_cache_options|
|enable_nz|bool|<ul><li>true</li><li>false</li></ul>|是否开启KV Cache NZ格式。<ul><li>仅DeepSeek-R1、DeepSeek-V3和DeepSeek-V3.1模型支持此特性。FA3量化场景下自动使能NZ格式。</li><li>默认值：false</li></ul>|
|weights_options|
|low_cpu_memory_mode|bool|<ul><li>true</li><li>false</li></ul>|是否开启低CPU内存占用模式。<ul><li>此特性需与Python组图同时开启。</li><li>仅Qwen2.5-7B模型支持此特性。</li><li>默认值：false（关闭）</li></ul><br>开启此功能后，权重加载阶段将逐Tensor加载模型参数，可显著降低CPU内存占用，尤其适用于边缘设备、小规格服务器等内存受限场景。在CPU内存充足的环境中，建议关闭该功能减少加载时间开销。|

## models参数

|配置项|取值类型|取值范围|配置说明|
|--|--|--|--|
|deepseekv2|map|-|deepseekv2相关配置。详情请参见[deepseekv2参数](#deepseekv2参数)。|

## deepseekv2参数 

|配置项|取值类型|取值范围|配置说明|
|--|--|--|--|
|ep_level|int|[1,2]|专家并行的实现形式。1：表示基于AllGather通信的EP并行<br>2：表示基于AllToAll和通算融合的EP并行|
|topk_scaling_factor|float|(0,1]|topk截断参数。<ul><li>“ep_level”=“1”时，每台设备的hidden_states后段部分为无效数据，可设置截断参数减小显存开销。</li><li>需同时配置 "enable_init_routing_cutoff"="true"。</li></ul>|
|enable_init_routing_cutoff|bool|<ul><li>true</li><li>false</li></ul>|是否允许topk截断。<ul><li>默认值：false（关闭）</li><li>“ep_level”=“1”时，可配置该参数。</li></ul>|
|alltoall_ep_buffer_scale_factors|list[list[int, float]]|列表每个成员包含两个数：第一个数为非负整数，第二个数为大于0的浮点数。<br>排列顺序按照第一个数的大小降序排列。|AllToAll通信buffer大小，第二层list包含两个元素，第一个数为序列长度、第二个数为buffer系数。序列长度为buffer系数的选择判断条件。示例：<br>[[1048576, 1.32], [524288, 1.4], [262144, 1.53], [131072, 1.8], [32768, 3.0], [8192, 5.2], [0, 8.0]]<ul><li>“ep_level”=“2”时，且用户需要精细化地管理显存的时候建议配置该项。</li><li>“ep_level”=“1”时该参数配置不生效。</li></ul>|
|num_dangling_shared_experts|int|正整数|共享专家外置的数量。<br>当前只支持Atlas 800I A3 超节点服务器 144卡且不开负载均衡的场景。建议配置成32。<br>默认值：0（关闭）|
|enable_mlapo_prefetch|bool|<ul><li>true</li><li>false</li></ul>|控制是否开启mlapo预取。<ul><li>true：开启</li><li>false：关闭</li></ul>默认值：false|
|enable_oproj_prefetch|bool|<ul><li>true</li><li>false</li></ul>|控制是否开启oproj预取。<br>Atlas 800I A2 推理服务器不建议开启。Atlas 800I A3 超节点服务器建议与OprojTp同时开启，推荐OprojTp设置为2。<ul><li>true：开启</li><li>false：关闭</li></ul>默认值：false|
|eplb|
|level|int|[0, 3]|<ul><li>0 : 不开启负载均衡</li><li>1 : 开启静态冗余负载均衡</li><li>2 : 开启动态冗余负载均衡（暂不支持）</li><li>3 : 开启强制负载均衡</li></ul>默认值：0|
|expert_map_file|string|该文件路径存在|静态冗余负载专家部署表路径。<br>默认值：""|
|num_redundant_experts|int|[0, n_routed_experts]|**当前版本暂不支持该参数。**<br>冗余专家的个数。<br>默认值：0|
|aggregate_threshold|int|≥1|**当前版本暂不支持该参数。**<br>表示动态EPLB算法触发的频率，单位是decode次数。<br>例如：50表示50次decode，触发一次动态EPLB算法，若算法认为热度超过一定阈值时，则调整路由表来降低算法热度。|
|buffer_expert_layer_num|int|[1, num_moe_layers]|**当前版本暂不支持该参数。**<br>表示动态EPLB每次搬运的layer个数。<br>由于权重搬运为异步搬运，在不影响原decode情况下，需要一个额外的buffer内存来存放被搬运中的新权重，配置为1层时，则为一次只搬运一层，然后刷新掉一层layer的权重和路由表。<br>影响的内存公式为：buffer_expert_layer_num*local_experts_num*44M (44M为一个int8的专家大小)|
|num_expert_update_ready_countdown|int|≥1|**当前版本暂不支持该参数。**<br>表示检查host->device搬运是否结束的频率，单位为decode次数。<br>因为搬运权重为异步搬运，必须所有ep卡搬运完毕后才能刷新权重和路由表，这里引入了通信，在搬运层较多的情况下，可以降低该频率，从而减少EPLB框架侧开销。|
|h3p|
|enable_qkvdown_dp|bool|<ul><li>true</li><li>false</li></ul>|控制是否开启qkvdown dp特性，减少计算及通信量，提升Prefill阶段性能。<br>默认值：“true”|
|enable_gating_dp|bool|<ul><li>true</li><li>false</li></ul>|控制是否开启gating dp特性，减少计算及通信量，提升Prefill阶段性能。<br>默认值：“true”<br>仅“ep_level”=“1”时，支持该特性。|
|enable_shared_expert_dp|bool|<ul><li>true</li><li>false</li></ul>|控制是否开启共享专家dp特性，提升Prefill阶段性能。<br>默认值：“false”<ul><li>仅“ep_level”=“1”时，支持该特性。</li><li>开启会占用额外显存从而可能产生“out of memory”的错误提示，建议设置为默认值。</li></ul>|
|enable_shared_expert_overlap|bool|<ul><li>true</li><li>false</li></ul>|控制是否开启共享专家的通信和计算双流掩盖特性，提升特定场景下（输入序列长度为2K~16K）的Prefill阶段性能。<br>默认值：“false”<ul><li>仅“ep_level”=“1”且“enable_shared_expert_dp”=“true”时支持该特性。</li><li>开启会占用额外显存从而可能产生“out of memory”的错误提示，建议设置为默认值。</li></ul>|
|enable_dispatch_combine_v2|bool|<ul><li>true</li><li>false</li></ul>|当“ep_level”=“2”时，控制是否开启dispatch算子和combine算子的v2版本，提升Decode阶段性能。<br>默认值：true|
|mix_shared_routing|bool|<ul><li>true</li><li>false</li></ul>|控制共享专家和路由专家是否合并的开关, 达到共享专家和路由专家并行计算的目的。<ul><li>不支持与CP特性叠加使用。</li><li>PD分离场景下，仅支持在D节点开启该开关。</li><li>默认值：false</li></ul>|
