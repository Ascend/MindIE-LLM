# Multi-Token Prediction (MTP)

MTP is a parallel decoding method used by DeepSeek to generate multiple tokens at a time. The core idea of MTP is that a model forecasts not just the subsequent token but several tokens concurrently during inference, which markedly enhances generation efficiency.

## Parameter Description

[Table 1](#table1) describes the parameters required for enabling the MTP feature.

**Table 1** MTP parameter in `ModelConfig` of `ModelDeployConfig` <a id="table1"></a>

|Parameter|Value Type|Value Range|Configuration Description|
|--|--|--|--|
|plugin_params|std::string|plugin_type: mtp<br>num_speculative_tokens: [1]|<ul><li>`plugin_type: mtp` indicates that MTP is enabled. </li><li>`num_speculative_tokens` indicates the number of MTP layers. The value can be `1` or `2`. </li><li>If no plugin function is required, remove this field from the configuration.</li></ul><br>Example: `{"plugin_type":"mtp","num_speculative_tokens": 1}`<br>**Note:** For the `num_speculative_tokens` setting: use `1` or `2` for low-latency scenarios; for high-throughput scenarios, it is advised to set no more than `1`.|

## Feature Combination

MTP can be used together with the following features:

1. Prefix cache and KV cache pooling
2. Asynchronous scheduling
3. KV cache int8 quantization
4. function call
5. Thinking analysis
6. Prefill-decode disaggregation (Both P and D nodes must be configured.)

MTP can be used together with the following features in some scenarios:

1. context_parallel
2. sequence_parallel

## Constraints

- The Atlas 800I A2 inference server and Atlas 800I A3 SuperPoD server support this feature.
- Only the W8A8 and KV cache INT8 quantization models of DeepSeek-R1 and DeepSeek-V3 support this feature.
- This feature supports W4A8 quantization.
- This feature cannot be used with parallel decoding, Multi-LoRA, or SplitFuse.
- When this feature is used together with context_parallel and sequence_parallel in the PD co-location scenario, num_speculative_tokens can only be set to 1.
- When context_parallel and sequence_parallel are used together with the MoE EP, only the P node supports context_parallel and sequence_parallel.
- This feature does not support postprocessing parameters related to multi-sequence inference, such as `n`, `best_of`, `use_beam_search`, and `logprobs`.
- MTP postprocessing supports only repetition penalty.

## Inference

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

2. Set serving parameters. Add the `plugin_params` field to the `config.json` file of the server. For details about the parameter fields, see [Table 1](#table1). For details about the serving parameters, see [Configuration Parameters (Serving)](../user_manual/service_parameter_configuration.md). The following is an example of parameter configuration.

    ```json
    "ModelDeployConfig" :
    {
       "maxSeqLen" : 2560,
       "maxInputTokenLen" : 2048,
       "truncation" : 0,
       "ModelConfig" : [
         {
             "plugin_params": "{\"plugin_type\":\"mtp\",\"num_speculative_tokens\": 1}",
             "modelInstanceType" : "Standard",
             "modelName" : "DeepSeek-R1_w8a8",
             "modelWeightPath" : "/data/weights/DeepSeek-R1_w8a8",
             "worldSize" : 8,
             "cpuMemSize" : 5,
             "npuMemSize" : -1,
             "backendType" : "atb",
             "trustRemoteCode" : false
          }
       ]
    },
    ```

3. Start the service.

   - **Installation using the `.whl` package:**

    ```bash
    mindie_llm_server
    ```

   - **Installation using the `.run` package:**

    ```bash
    ./bin/mindieservice_daemon
    ```
