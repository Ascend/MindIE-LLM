# Prefix Cache

Currently, the KV cache mechanism is widely used in LLM inference systems. However, this mechanism has the following problems:

1. As sequence lengths supported by LLMs continuously increase, the graphics memory resources required by the KV cache also sharply increase.
2. The KV cache is valid only for the current session. If duplicate token sequences exist across sessions, the KV cache cannot be reused.

Prefix cache stores the KV cache of completed sessions in a hash table. For a new session request, the system checks whether the same token sequence exists in the table. If yes, the previously computed KV cache can be reused across sessions.

Advantages:

- **Shorter prefill time**: By reusing the KV cache corresponding to the repeated cross-session token sequence, the computation time for some prefix tokens can be reduced, thereby decreasing the prefill time.
- **More efficient graphics memory usage**: When the sessions being processed have a common prefix, the KV cache of the common prefix can be shared, reducing redundant graphics memory usage.

## Constraints

- The Atlas 800I A2 inference server, Atlas 300I Duo inference card, and Atlas 800I A3 SuperPoD server support this feature.
- The Qwen2 series, Qwen2.5 series, Qwen3 series, DeepSeek-R1, and DeepSeek-V3/V3.1 models support this feature.
- The KV cache of the public prefix tokens is reused only when the number of cross-session public prefix tokens is greater than or equal to the block size.
- Prefix cache supports only W4A8 quantization, W8A8 quantization, PDMIX quantization, and sparse quantization.
- This feature cannot be used with Multi-LoRA.
- This feature can be used with prefill-decode disaggregation, parallel decoding, MTP, KV cache pooling, asynchronous scheduling, SplitFuse, context parallel + sequence parallel, and C8 quantization.
- This feature supports the `n`, `best_of`, and `use_beam_search` postprocessing parameters.
- In the prefill-decode disaggregation scenario, this feature needs to be enabled only on the prefill node.
- You are advised not to enable this feature when the prefix reuse rate is low or no prefix is reused.
- The combination of prefix cache, context parallel, sequence parallel, and function call (multiturn) is not supported.

## Parameter Description

[Table 1](#table1), [Table 2](#table2), and [Table 3](#table3) list the supplementary parameters required for enabling the Prefix Cache feature.

**Table 1** Prefix Cache parameter (1): `ModelConfig` in `ModelDeployConfig` <a id="table1"></a>

|Parameter|Value Type|Value Range|Configuration Description|
|--|--|--|--|
|plugin_params|std::string|"{\"plugin_type\":\"prefix_cache\"}"|<ul><li>If the value is set to `{"plugin_type":"prefix_cache"}`, Prefix Cache is executed. </li><li>If no plugin function is required, remove this field from the configuration.</li></ul>|

**Table 2** Prefix Cache parameter (2): `ScheduleConfig` <a id="table2"></a>

|Parameter|Value Type|Value Range|Configuration Description|
|--|--|--|--|
|enablePrefixCache|-|-|This field is no longer required. The configuration in the current version does not affect the earlier version.<br>This field is expected to be deleted in Q1 of 2026.|

**Table 3** Prefix Cache parameters (3): `models` in `ModelConfig` <a id="table3"></a>

|Parameter|Value Type|Value Range|Configuration Description|
|--|--|--|--|
|**deepseekv2**|-|-|-|
|**kv_cache_options**|-|-|-|
|enable_nz|bool|<ul><li>true</li><li>false</li></ul>|Specifies whether to enable the NZ format for the KV cache.<br><ul><li>Only the DeepSeek-R1, DeepSeek-V3, and DeepSeek-V3.1 models support this feature. The NZ format is automatically enabled in the FA3 quantization scenario. </li><li>This format must be enabled for the DeepSeek-R1, DeepSeek-V3, and DeepSeek-V3.1 models. For other models, it is disabled. </li><li>Default value: `false`</li></ul>|

## Inference

The following uses multi-round dialog as an example to describe how to use the prefix cache feature.

1. Open the `config.json` file of the server.

   - **Installation using the `.whl` package:**

    ```bash
    cd {MindIE_installation_directory}/mindie_llm/
    vi conf/config.json
    ```

    - **Installation using the `.run` package:**

    ```bash
    cd {MindIE_installation_directory}/latest/mindie-service
    vi conf/config.json
    ```

2. Set serving parameters. Add parameters to the `config.json` file of the server according to [Table 1](#table1) to [Table 3](#table3). For details about the serving parameters, see [Configuration Parameters (Serving)](../user_manual/service_parameter_configuration.md). The following is an example of parameter settings.

    The following uses the DeepSeek-R1 model as an example to describe how to enable the prefix cache feature.

    ```json
    "ModelDeployConfig" :
    {
       "maxSeqLen" : 2560,
       "maxInputTokenLen" : 2048,
       "truncation" : 0,
       "ModelConfig" : [
         {
             "plugin_params": "{\"plugin_type\":\"prefix_cache\"}",
             "modelInstanceType" : "Standard",
             "modelName" : "DeepSeek-R1_w8a8",
             "modelWeightPath" : "/data/weights/DeepSeek-R1_w8a8",
             "worldSize" : 8,
             "cpuMemSize" : 5,
             "npuMemSize" : -1,
             "backendType" : "atb",
             "trustRemoteCode" : false,
             "models": {
                 "deepseekv2": {
                     "kv_cache_options": {"enable_nz": true}
                  }
             }
          }
       ]
    },
    ```

    > [!NOTE]
    >- For non-DeepSeek models, the `models` field is not required.
    >- If multiple features need to be used together, such as prefix cache and MTP, separate the feature names with commas (,) as follows:
    > **"plugin_params": "{\"plugin_type\":\"mtp,prefix_cache\",\"num_speculative_tokens\": 1}"**,

3. Start the service.

   - **Installation using the `.whl` package:**

    ```bash
    mindie_llm_server
    ```

   - **Installation using the `.run` package:**

    ```bash
    ./bin/mindieservice_daemon
    ```

4. Send a request for the first time. The prompt is the first round of questions.

    ```bash
    curl https://127.0.0.1:1025/generate \
    -H "Content-Type: application/json" \
    --cacert ca.pem --cert client.pem  --key client.key.pem \
    -X POST \
    -d '{
    "inputs": "Question: Parents have complained to the principal about bullying during recess. The principal wants to quickly resolve this, instructing recess aides to be vigilant. Which situation should the aides report to the principal?\na) An unengaged girl is sitting alone on a bench, engrossed in a book and showing no interaction with her peers.\nb) Two boys engaged in a one-on-one basketball game are involved in a heated argument regarding the last scored basket.\nc) A group of four girls has surrounded another girl and appears to have taken possession of her backpack.\nd) Three boys are huddled over a handheld video game, which is against the rules and not permitted on school grounds.\nAnswer:",
    "parameters": {"max_new_tokens":512}
    }'
    ```

5. Send a second request with the prompt in the format of "first-round question + first-round answer + second-round question". In this case, the first-round question is a reusable public prefix. (The actual reused part may not be the complete prompt of the first-round question. The cache is implemented in the unit of block, and the prefix cache is stored in a multiple of `blocksize`. For example, if the number of tokens in the first-round question is 164, and `blocksize` is `128`, only the first 128 tokens are reused.)

    ```bash
    curl https://127.0.0.1:1025/generate \
    -H "Content-Type: application/json" \
    --cacert ca.pem --cert client.pem  --key client.key.pem \
    -X POST \
    -d '{
    "inputs": "Question: Parents have complained to the principal about bullying during recess. The principal wants to quickly resolve this, instructing recess aides to be vigilant. Which situation should the aides report to the principal?\na) An unengaged girl is sitting alone on a bench, engrossed in a book and showing no interaction with her peers.\nb) Two boys engaged in a one-on-one basketball game are involved in a heated argument regarding the last scored basket.\nc) A group of four girls has surrounded another girl and appears to have taken possession of her backpack.\nd) Three boys are huddled over a handheld video game, which is against the rules and not permitted on school grounds.\nAnswer:c) A group of four girls has surrounded another girl and appears to have taken possession of her backpack.\nExplanation: The principal wants to quickly resolve this, instructing recess aides to be vigilant. The principal is concerned about bullying during recess. The principal wants the aides to report any bullying behavior to him. The principal is not concerned about the other situations.\nQuestion: If the aides confront the group of girls from situation (c) and they deny bullying, stating that they were merely playing a game, what specific evidence should the aides look for to determine if this is a likely truth or a cover-up for bullying?\nAnswer:",
    "parameters": {"max_new_tokens":512}
    }'
    ```
