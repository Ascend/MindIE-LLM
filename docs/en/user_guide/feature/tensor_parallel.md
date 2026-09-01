# Tensor Parallel

Tensor parallelism (TP) is a model parallelism strategy that splits tensors (such as weight matrices and activation values) among multiple devices (such as NPUs) to implement distributed model inference.

## Constraints

- The Atlas 800I A2 inference server and Atlas 800I A3 SuperPoD server support this feature.
- DeepSeek-V3 and DeepSeek-R1 support local TP splitting of the LmHead matrix, local TP splitting of the O project matrix, and TP greater than 1.
- Prefill-decode disaggregation scenarios with distributed decode nodes support local TP splitting of the LmHead and O project matrices, which reduces the matrix computation time and inference latency.
- In prefill-decode disaggregation scenarios with distributed, low-latency decode nodes, if TP exceeds 1, TP splitting of MLA is supported, which reduces the decode inference latency in small-batch and low-latency scenarios.
- If `tp` exceeds 1, this feature cannot be enabled together with local TP splitting of the O project matrix, and you are not advised to enable this feature together with local TP splitting of the LmHead matrix.

## Parameter Description

[Table 1](#table1) describes the parameters required for enabling local TP splitting of the LmHead matrix.

**Table 1** Parameter for local TP splitting of the LmHead matrix: `models` in `ModelConfig` <a id="table1"></a>

|Parameter|Value Type|Value Range|Configuration Description|
|--|--|--|--|
|**deepseekv2**|-|-|-|
|**parallel_options**|-|-|-|
|lm_head_local_tp|int|[1, `worldSize`/Number of nodes]|TP split count for LmHead.<br><ul><li>Only the DeepSeek-R1, DeepSeek-V3, and DeepSeek-V3.1 models support this feature. </li><li>Default value: `-1`, indicating that splitting is disabled. </li></ul>|

[Table 2](#table2) describes the parameters required for enabling local TP splitting of the O project matrix.

**Table 2** Parameter for local TP splitting of the O project matrix: `models` in `ModelConfig`
<a id="table2"></a>

|Parameter|Value Type|Value Range|Configuration Description|
|--|--|--|--|
|**deepseekv2**|-|-|-|
|**parallel_options**|-|-|-|
|o_proj_local_tp|int|[1, worldSize / Number of nodes]|Split count for the Attention O matrix.<br><ul><li>Only the DeepSeek-R1, DeepSeek-V3, and DeepSeek-V3.1 models support this feature. </li><li>The default value is `-1`, indicating that sharding is disabled. </li></ul>|

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

2. Set serving parameters. Add parameters to the `config.json` file of the server based on [Table 1](#table1) and [Table 2](#table2). For details about the serving parameters, see [Configuring Parameters (serving)](../user_manual/service_parameter_configuration.md). The following is an example of parameter configuration.

    The following uses the DeepSeek-R1 model as an example. In addition, enabling TP splitting and disabling local TP splitting of the LmHead and O project matrices are used as examples.

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
             "worldSize" : 8,
             "cpuMemSize" : 5,
             "npuMemSize" : -1,
             "backendType" : "atb",
             "trustRemoteCode" : false,
             "tp": 2,
             "models": {
                "deepseekv2": {
                    "parallel_options": {
                        "lm_head_local_tp": -1,
                        "o_proj_local_tp": -1
                    }
                }
             }
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
