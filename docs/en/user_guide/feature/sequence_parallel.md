# Sequence Parallel

Sequence parallelism (SP) splits the KV cache so that the KV cache saved by each SP rank is different, reducing the graphics memory usage and supporting long sequences.

## Constraints

- The Atlas 800I A2 inference server and Atlas 800I A3 SuperPoD server support this feature.
- Currently, only the W8A8 quantization models of DeepSeek-R1, W4A8 quantization models of DeepSeek-R1, DeepSeek-V3, and DeepSeek-V3.1 support this feature.
- This feature is supported in the prefill-decode disaggregation and prefill-decode co-location scenarios.
- The value of SP must be equal to that of TP.
- In the prefill-decode co-location scenario:
    - This feature can be used together with DP and TP. The product of DP and TP is equal to the value of `worldSize`.
    - This feature can be used together with CP, TP, and MTP. The product of CP and TP is equal to `worldSize`.
    - This feature can be used together with asynchronous scheduling and prefix cache, and it can be used in scenarios where MTP equals 1.

- In the prefill-decode disaggregation scenario:
    - SP can be enabled on prefill nodes only. This feature can be used together with DP, TP, and MTP. The product of DP and TP is equal to the value of `worldSize`.
    - SP can be enabled on prefill nodes only. This feature can be used together with CP, TP, and MTP. The product of CP and TP is equal to the value of `worldSize`.
    - This feature can be used together with MTP, asynchronous scheduling, and Prefix Cache.

- This feature does not support BF16.

## Parameter Description

[Table 1](#table1) describes the service parameters required for enabling the SP feature.

**Table 1** SP parameter `ModelConfig` in `ModelDeployConfig` <a id="table1"></a>

|Parameter|Value Type|Value Range|Configuration Description|
|--|--|--|--|
|sp|int|sp=tp|Number of parts obtained after KV cache splitting.|

## Inference

1. Open the `config.json` file of the server.

    ```bash
    cd {MindIE_installation_directory}/mindie_llm/
    vi conf/config.json
    ```

2. Set serving parameters. Add the "sp" field to the `config.json` file of the server. For details about the parameters, see [Table 1](#table1). For detailed configuration of the `config.json` file: refer to "Cluster Service Deployment" \> "PD Disaggregation" in *MindIE Motor Development Guide* for PD disaggregation and refer to "Configuring MindIE" \> "Configuring Server" > "Multi-Node Inference" in *MindIE Installation Guide* for PD co-location.

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
                "dp": 2,
                "sp": 8,
                "tp": 8,
                "moe_ep": 16,
                "moe_tp": 1
            }
        ]
    }
    ```

3. Start the service.

    ```bash
    mindie_llm_server
    ```
