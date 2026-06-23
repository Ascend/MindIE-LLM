# SLO Scheduling Tuning

An SLO defines a target value for a specific metric over a certain period of time. To handle high-concurrency client requests and improve system throughput while meeting SLO requirements, the following approaches are provided:

1. Prefill/Decode phase selection algorithm based on TTFT/TPOT latency prediction and the Least Laxity First (LLF) algorithm.
This algorithm collects TTFT and TPOT latency data for fitting modeling to predict the execution time of each prefill or decode phase, and uses the LLF algorithm to determine whether prefill or decode is executed for the next batch. It is suitable for scenarios with strict requirements on both TTFT and TPOT, enabling higher throughput under high-concurrency workloads while meeting SLO requirements.

2. Dynamic batch size adjustment algorithm based on real-time TPOT awareness.
This algorithm continuously monitors the system TPOT latency and compares it with the SLO-defined decode latency target. Depending on the comparison result, `maxPrefillBatchSize` and `maxBatchSize` are dynamically adjusted to prevent all requests from being loaded into on-chip memory, which could cause system congestion and degrade throughput. This algorithm is suitable for scenarios with strict requirements on TPOT, prioritizing responses to requests already loaded into on-chip memory under high-concurrency workloads. Due to real-time fluctuations in TPOT data collection, the actual latency may deviate by roughly 10% from the configured target.

## Constraints

- Only the Atlas 800I A2 inference server supports this feature.
- DeepSeek-R1, DeepSeek-V3, and Qwen series models support this feature.
- This feature applies only to PD co-location and cannot be enabled together with the SplitFuse feature.
- This feature provides significant benefits for short outputs (less than 256 tokens). As the output length increases, the throughput gain decreases.

## Parameter Description

[Table 1](#table1) describes the parameters required for enabling the SLO scheduling tuning feature.

**Table 1** Parameters of the SLO scheduling tuning feature<a id="table1"></a>

|Parameter|Value Type|Value Range|Configuration Description|
|--|--|--|--|
|stageSelectPolicy|uint32_t|[0,2]|Prefill/Decode selection policy.<br><ul><li>`0`: Prioritize prefill.</li><li>`1`: Prioritize throughput.</li><li>`2`: Select PD phases based on TTFT/TPOT latency prediction and LLF algorithm.</li></ul><br>Optional. The default value is `0`.|
|dynamicBatchSizeEnable|bool|<ul><li>true</li><li>false</li></ul>|Specifies whether to enable the dynamic batch size adjustment algorithm.<br>Optional. The default value is `false`.|
|prefillExpectedTime|uint32_t|[0,10000]|Expected SLO latency during token generation in the prefill phase.<br>Optional. The default value is `1500`.|
|decodeExpectedTime|uint32_t|[0,10000]|Expected SLO latency during token generation in the decode phase.<br>Optional. The default value is `50`.|

## Inference

This section describes how to use the SLO tuning optimization function.

1. Open the `config.json` file of MindIE Motor.

    ```bash
    cd {MindIE_installation_directory}/mindie_llm/
    vi conf/config.json
    ```

2. Set serving parameters. Add the `stageSelectPolicy`, `dynamicBatchSizeEnable`, `prefillExpectedTime`, and `decodeExpectedTime` fields (the following bold parts) to the `config.json` file of the server. For details about the parameter fields, see [Table 1](#table1). For details about the serving parameters, see [Configuration Parameters (Serving)](../user_manual/service_parameter_configuration.md). The following is an example of parameter configuration.

    ```json
    "stageSelectPolicy" : 2,
    "dynamicBatchSizeEnable" : true,
    "prefillExpectedTime" : 1000,
    "decodeExpectedTime" : 50
    ```

3. Start the service.

    ```bash
    mindie_llm_server
    ```

4. Start tuning. This example uses the AISBench tool and GSM8K dataset, with concurrency set to `500`. The AISBench tool is configured as follows. For details, see "[Performance Test](../quick_start/quick_start.md#performance-test)" in *Quick Start*.

    ```text
    models = [
        dict(
            attr="service",
            type=VLLMCustomAPIChatStream,
            abbr='vllm-api-stream-chat',
            path="$ModelPath",
            model="$ModelName",
            request_rate = $1,
            retry = 2,
            host_ip = "{ipAddress}",
            host_port = "{port}",
            max_out_len = 64,
            batch_size= 500,
            trust_remote_code=False,
            generation_kwargs = dict(
                temperature = 0,
                ignore_eos = True
            ),
            pred_postprocessor=dict(type=extract_non_reasoning_content)
        )
    ]

    ```
