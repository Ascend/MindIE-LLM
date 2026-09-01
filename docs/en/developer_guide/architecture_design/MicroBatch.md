# Micro Batch

Micro-batch processing is a technique where data is split into smaller batches for execution. In the current implementation, an additional data stream is created to split a batch of data into two batches, which are executed on two separate data streams. When data stream 1 performs computation, data stream 2 performs communication. The communication–computation overlap enables hardware resources to be fully utilized to improve inference throughput.

Data streams are synchronized using the event mechanism, ensuring that computation and communication tasks do not conflict with each other and preventing hardware resource contention. This feature is typically used in the prefill phase because communication operators consume long execution time and the execution duration of communication and compute operators is balanced. In this implementation, the overlap between computation and communication exceeds 70%.

## Constraints

- This feature cannot be enabled together with the communication-computing fused operator feature.
- This feature cannot be enabled together with the Python graph.
- This feature can be enabled only together with the quantization feature.
- Only the Qwen2.5-14B, Qwen3-14B, DeepSeek-R1, and DeepSeek-V3.1 models support this feature.
- Enabling this feature will occupy extra graphics memory.
- In serving scenarios, if the number of KV caches decreases, scheduling will be affected and the throughput will decrease. Therefore, you are advised not to enable this feature when the graphics memory is limited.

## Parameter Description

**Table 1** describes the parameters required for enabling the micro batch feature.

### Table 1 Micro Batch parameter: `models` in `ModelConfig`

| Configuration Item| Value| Value Range| Configuration Description|
|--------|----------|----------|----------|
| `stream_options` → `micro_batch` | `bool` | `true` / `false` | Whether to enable the communication-computing dual-stream overlapping feature.<br>Default value: `false` (disabled)|

## Inference

1. Open the `config.json` file of the server.

    ```bash
    cd {MindIE_installation_directory}/mindie_llm/
    vi conf/config.json
    ```

2. Set serving parameters. Add the `"micro_batch"` field to the Server `config.json` file as shown in the example below. For parameter descriptions, refer to Table 1 Micro Batch parameter: `models` in `ModelConfig`. For serving parameters, see "Configuration Parameters (Serving)". Below is an example configuration:

    ```json
    "ModelDeployConfig": {
      "maxSeqLen": 2560,
      "maxInputTokenLen": 2048,
      "truncation": 0,
      "ModelConfig": [
        {
          "modelInstanceType": "Standard",
          "modelName": "Qwen3-14B",
          "modelWeightPath": "/data/weights/Qwen3-14B",
          "worldSize": 8,
          "cpuMemSize": 5,
          "npuMemSize": -1,
          "backendType": "atb",
          "trustRemoteCode": false,
          "models": {
            "qwen3": {
              "ccl": {
                "enable_mc2": false
              },
              "stream_options": {
                "micro_batch": true
              }
            }
          }
        }
      ]
      }
    ```

3. Start the service.

    ```bash
    mindie_llm_server
    ```
