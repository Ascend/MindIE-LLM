# Context Parallel

Context parallelism (CP) performs parallel computing for the self-attention module in the sequence dimension. CP splits long sequences in the context dimension, allocates the sequences to different devices for parallel processing, and reduces the response time of the first token. The CP implementation includes the following:

1. Each device calculates its own attention, and devices transfer KV values in ring mode to obtain the result of the block-based computation. The overall principle is similar to ring-attention.
2. The flash-attention 2 algorithm is used to perform block-based computation and correct the result.

## Constraints

- The Atlas 800I A2 inference server and Atlas 800I A3 SuperPoD server support this feature.
- Currently, only the W8A8 quantization models of DeepSeek-R1, W4A8 quantization models of DeepSeek-R1, DeepSeek-V3, and DeepSeek-V3.1 support this feature.
- Currently, CP cannot be enabled independently. To enable CP, sequence parallelism (SP) must be enabled at the same time.
- This feature is supported in the prefill-decode disaggregation and prefill-decode co-location scenarios.
- In the prefill-decode co-location scenario:
  - This feature can be used together with SP and tensor parallelism (TP). When CP is enabled, the value of DP must be 1, the value of SP must be equal to that of TP, and the product of CP, DP, and TP must be equal to the value of `worldSize`.
  - This feature supports combined use with MTP=1, asynchronous scheduling, and Prefix Cache.

- In the prefill-decode disaggregation scenario:
  - CP can be enabled on prefill nodes only. This feature can be used together with SP, TP, and MTP. When CP is enabled, the value of DP must be 1, the value of SP must be equal to that of TP, and the product of CP, DP, and TP must be equal to the value of `worldSize`.
  - This feature can be used together with MTP, asynchronous scheduling, and Prefix Cache.

- This feature does not support BF16.

## Description

[Table 1](#ModelConfig parameters) describes the serving parameters required for enabling the CP feature.

**Table 1** ModelConfig in ModelDeployConfig <a id="ModelConfig parameters"></a>

|Parameter|Value Type|Value Range|Configuration Description|
|--|--|--|--|
|cp|int|[1, 2]|Number of parts obtained after an input sequence is split.<br>**1**: indicates that the CP feature is disabled.<br>**2**: indicates that the input sequence is split into two parts.<br>Currently, if the CP feature is enabled, the number of split parts can only be 2.|

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

2. Set serving parameters. Add the `cp` field (the following information in bold) to the `config.json` file of the server. For details about the parameters, see [Parameter Description](#description). For details about the serving parameters, see [Configuration Parameters (Serving)](../user_manual/service_parameter_configuration.md). The following is an example of parameter settings.

    ```json
    "ModelDeployConfig" :
    {
        "maxSeqLen" : 2560,
        "maxInputTokenLen" : 2048,
        "truncation" : 0,
        "ModelConfig" : [
            {
                "modelInstanceType" : "Standard",
                "modelName" : "DeepSeek-R1_w8a8",
                "modelWeightPath" : "/data/weights/DeepSeek-R1_w8a8",
                "worldSize" : 16,
                "cpuMemSize" : 5,
                "npuMemSize" : -1,
                "backendType" : "atb",
                "trustRemoteCode" : false,
                "dp": 1,
                "cp": 2,
                "sp": 8,
                "tp": 8,
                "moe_ep": 16,
                "moe_tp": 1
            }
        ]
    }
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
