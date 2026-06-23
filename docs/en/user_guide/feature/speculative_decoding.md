# Parallel Decoding

In LLM inference scenarios, conventional auto-regressive decoding is inherently slow due to its step-by-step nature, which restricts concurrency. Although the inference phase is constrained by memory bandwidth, it often has excessive computing resources. To address this imbalance, parallel decoding introduces speculative execution—an optimization technique commonly used in processor architectures—that leverages the excessive computing resources to improve concurrency. However, enabling parallel decoding requires the prompt input to retain a trie-tree and a draft token map, which affects the TTFT.

Advantages of parallel decoding:

In small-batch inference scenarios—such as those involving sufficiently long inputs/outputs or code generation—parallel decoding can offset limited memory bandwidth by utilizing excess computing resources, thereby enhancing computing efficiency. The effectiveness of parallel decoding is closely tied to the ratio of validated tokens. As a result, greedy decoding offers the greatest benefit, while sampling and penalty mechanisms may reduce its impact.

To fully leverage parallel decoding, the following conditions should be met:

1. A low number of concurrent requests, constrained memory bandwidth, and surplus computational resources.
2. Sufficiently long input to provide an initial source of candidate tokens.
3. Extended output length, allowing parallel decoding to reduce inference steps and deliver performance gains.

Two parallel decoding algorithms are supported, distinguished by their respective methods of candidate token generation, as shown in [Table 1](#table1).

**Table 1** Parallel decoding algorithms<a id="table1"></a>

|Parallel decoding algorithms|Candidate Token Generation|Application Scenario|
|--|--|--|
|memory_decoding|Uses a trie-tree to cache historical inputs and outputs of a model and obtain candidate tokens.|Code generation or retrieval|
|lookahead|Generates candidate tokens based on Jacobi iteration, prompts, and output results.|Text generation, dialog systems, and diversified query answering|

## Constraints

- The Atlas 800I A2 inference server and Atlas 300I Duo inference card support this feature.
- Only the Llama3 series, Qwen2 series, Qwen2.5 series, Qwen3-14B, and Qwen3-32B models support this feature.
- Parallel decoding supports only W8A8 quantization and sparse quantization.
- This feature cannot be used with prefill-decode disaggregation, Multi-LoRA, SplitFuse, long sequence, MTP, asynchronous scheduling, or multi-server inference.
- Currently, this feature does not support postprocessing parameters related to multi-sequence inference, such as `n`, `best_of`, `use_beam_search`, `logprobs`, and `top_logprobs`.
- Streaming inference is not supported in parallel decoding scenarios.
- Parallel decoding penalty postprocessing supports only repetition penalty.
- Health check is not supported in parallel decoding scenarios.
- The lookahead and memory\_decoding algorithms cannot be enabled at the same time.

## Parameter Description

[Table 2](#table2) to [Table 6](#table6) describe the parameters required for enabling the parallel decoding feature.

**Table 2** memory\_decoding parameter: `ModelConfig` in `ModelDeployConfig <a id="table2"></a>

|Parameter|Value Type|Value Range|Configuration Description|
|--|--|--|--|
|plugin_params|std::string|plugin_type: memory_decoding<br>decoding_length: [1, 16]<br>dynamic_algo: true or false|<ul><li>If `plugin_type` is set to `memory_decoding`, memory_decoding is used for parallel decoding. </li><li>`decoding_length` is a parameter in the memory_decoding algorithm and indicates the maximum length of candidate tokens. The default value is `16`. </li><li>`dynamic_algo` is an optional parameter. If it is set to `true`, the dynamic candidate length adaptation function is enabled. The default value is `false`. </li><li>If no plugin function is required, remove this field from the configuration. </li><li>Examples:`{"plugin_type":"memory_decoding","decoding_length": 16,"dynamic_algo": true}`· or `{"plugin_type":"memory_decoding","decoding_length": 16}`</li></ul>|

**Table 3** memory\_decoding parameter: `ModelDeployConfig`

|Parameter|Value Type|Value Range|Configuration Description|
|--|--|--|--|
|speculationGamma|uint32_t|Related to plugin parameters|In memory_decoding mode, the value of this field must be greater than or equal to that of **decoding_length**.<br>It is recommended that the value be equal to **decoding_length**.|

**Table 4** memory\_decoding parameter: `ScheduleConfig`

|Parameter|Value Type|Value Range|Configuration Description|
|--|--|--|--|
|maxIterTimes|uint32_t|Related to plugin parameters|If `dynamic_algo` is set to `true`, the value must be greater than or equal to the expected output length plus the value of `speculationGamma`.<br>For example, if the expected maximum output length is 512, the value must be greater than or equal to 512 + `speculationGamma`.|

**Table 5** Lookahead parameter: `ModelConfig` in `ModelDeployConfig`

|Parameter|Value Type|Value Range|Configuration Description|
|--|--|--|--|
|plugin_params|std::string|plugin_type: la<br>level: [3, 16]<br>window: [1, 16]<br>guess_set_size: [1, 16]|If `plugin_type` is set to `la`, lookahead is used for parallel decoding.<br>The `level`, `window`, and `guess_set_size` parameters correspond to the N, W, and G parameters in the lookahead algorithm, respectively. The default values are `4`, `5`, and `5`, and the maximum value for each parameter is `16`. Example: `"{"plugin_type":"la","level": 4,"window": 5,"guess_set_size": 5}"`|

**Table 6** Lookahead parameter: `ModelDeployConfig` <a id="table6"></a>

|Parameter|Value Type|Value Range|Configuration Description|
|--|--|--|--|
|speculationGamma|uint32_t|Related to plugin parameters|In lookahead, the value must be greater than or equal to (N – 1) x (W + G).<br>Recommended value: (N – 1) x (W + G)|

## Inference <a name="section1788515529541"></a>

1. Open the `config.json` file of the server.

    ```bash
    cd {MindIE_installation_directory}/mindie_llm/
    vi conf/config.json
    ```

2. Set serving parameters. Add parameters to the `config.json` file of the server according to [Table 2](#table2) to [Table 6](#table6). For details about the serving parameters, see [Configuration Parameters (Serving)](../user_manual/service_parameter_configuration.md). The following is an example of parameter configuration.

    Example of parallel decoding configuration for the memory\_decoding algorithm:

    ```json
    "ModelDeployConfig" :
    {
        "maxSeqLen" : 2560,
        "maxInputTokenLen" : 2048,
        "truncation" : 0,
        "speculationGamma": 16,
        "ModelConfig" : [
            {
                "plugin_params":"{\"plugin_type\":\"memory_decoding\",\"decoding_length\":16,\"dynamic_algo\":true}",
                "modelInstanceType" : "Standard",
                "modelName" : "llama3-70b",
                "modelWeightPath" : "/data/weights/llama3-70b",
                "worldSize" : 4,
                "cpuMemSize" : 5,
                "npuMemSize" : -1,
                "backendType" : "atb",
                "trustRemoteCode" : false
            }
        ]
    }
    ```

    Configuration example of the lookahead algorithm:

    ```json
    "ModelDeployConfig" :
    {
        "maxSeqLen" : 2560,
        "maxInputTokenLen" : 2048,
        "truncation" : 0,
        "speculationGamma": 30,
        "ModelConfig" : [
            {
                "plugin_params":"{\"plugin_type\":\"la\",\"level\":4,\"window\":5,\"guess_set_size\":5}",
                "modelInstanceType" : "Standard",
                "modelName" : "Qwen2.5-7B-Instruct",
                "modelWeightPath" : "/data/weights/Qwen2.5-7B-Instruct",
                "worldSize" : 1,
                "cpuMemSize" : 5,
                "npuMemSize" : -1,
                "backendType" : "atb",
                "trustRemoteCode" : false
            }
        ]
    }
    ```

3. Start the service.

    ```bash
    mindie_llm_server
    ```
