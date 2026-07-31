# Data Parallel

Data parallelism (DP) splits inference requests into multiple batches and allocates them to different compute devices for parallel processing. These devices process different batches of data in parallel, and then merge the results.

## Scenario

When the graphics memory is sufficient, the data parallel feature can be enabled to improve the throughput.

## Constraints

- The Atlas 800I A2 inference server and Atlas 800I A3 SuperPoD server support this feature.
- The Attention and MLP modules of all models support this feature.
- DP can be used together with tensor parallelism in the same module.

## Description

[Table 1](#table1) describes the supplementary parameters required for enabling the data parallel feature.

**Table 1** `ModelConfig` in `ModelDeployConfig` <a id="table1"></a>

|Parameter|Value Type|Value Range|Configuration Description|
|--|--|--|--|
|tp|int32_t|<ul><li>If `dp` is not set or is set to `-1`, the value is identical to that of the `worldSize` parameter. </li><li>When used together with `dp`, the value of `tp * dp` must be equal to that of the `worldSize` parameter.</li></ul><br>For example, if `worldSize` is set to `8` and `dp` is set to `2`, `tp` must be set to `4`.|Number of tensor parallelism processes on the entire network.<br>(Optional) The default value is the value of `worldSize`.|
|dp|int32_t|<ul><li>When this parallelism mode is not used: `-1`</li><li> When used together with `tp`, the value of `dp * tp` must be equal to that of the `worldSize` parameter.</li></ul><br>For example, if `worldSize` is set to `8` and `tp` is set to `4`, `dp` must be set to `2`.|Number of DP processes in the Attention module.<br>(Optional) The default value is `-1`, indicating that data parallelism is not performed.|
|cp|int32_t|<ul><li>When this parallelism mode is not used: `1`</li><li> When used together with `sp`, the value of `dp * tp * cp` must be equal to that of the `worldSize` parameter, and `dp` must be set to `1`.</li></ul><br>For example, if `worldSize` is set to `16`, `tp` is set to `8`, and `sp` is set to `8`, `dp` must be set to `1` and `cp` must be set to `2`.|(Optional) The default value is `1`, indicating that context parallelism is not performed.<br>Number of context parallelism processes in the Attention module.|
|sp|int32_t|<ul><li>When this parallelism mode is not used: `1`</li><li>When used together with `tp`, the value of `sp` must be equal to that of `tp`.</li></ul><br>For example, if `worldSize` is set to `16`, `tp` is set to `8`, and `dp` is set to `2`, `sp` must be set to `8`.|(Optional) The default value is `1`, indicating that sequence parallelism is not performed.<br>Number of sequence parallelism processes in the Attention module.|

> [!NOTE]
> If the preceding supplementary parameters are not set, the `tp` and `moe_tp` parallelism modes are used by default during inference.

## Inference

CANN and MindIE have been installed in the environment. For details, see *MindIE Installation Guide*.

1. Set environment variables for optimizing graphics memory allocation.

    ```bash
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
    ```

2. Open the `config.json` file of the server.

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

3. Set serving parameters. Add parameters to the `config.json` file of the server according to [Table 1](#table1). For details about the serving parameters, see [Configuration Parameters (Serving)](../user_manual/service_parameter_configuration.md). The following is an example of parameter settings.

    ```json
    "ModelConfig" : [
        {
            "modelInstanceType" : "Standard",
            "modelName" : "deepseekv2",
            "modelWeightPath" : "/home/data/DeepSeek-V2-Chat-W8A8-BF16/",
            "worldSize" : 8,
            "cpuMemSize" : 5,
            "npuMemSize" : 1,
            "backendType" : "atb",
            "trustRemoteCode" : false,
            "tp": 1,
            "dp": 8,
            "cp": 1,
            "sp": 1
        }
    ]
    ```

    In the preceding parameter settings, eight devices are used for inference, the Attention module uses DP, and the MoE model does not use tensor parallelism.

4. Start the service.

   - **Installation using the `.whl` package:**

    ```bash
    mindie_llm_server
    ```

   - **Installation using the `.run` package:**

    ```bash
    ./bin/mindieservice_daemon
    ```

5. Send an inference request. For details, see "Cluster Management Components" \> "Coordinator" \> "RESTful APIs" \> "User-Side APIs" \> "OpenAI Inference APIs" in *MindIE Motor Development Guide*.
