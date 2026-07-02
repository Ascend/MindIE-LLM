# Thinking Analysis

Some LLMs include the thinking process in their outputs. This feature is designed to structurally parse the output, separating the model's thinking process (think) from the final output (content) and storing them in the `reasoning_content` and `content` fields, respectively.

- `reasoning_content`: stores the model's internal reasoning, analysis, and logic judgment before generating the final answer.
- `content`: stores the model's final output answer or decision.

## Constraints

- The Atlas 800I A2 inference server, Atlas 800I A3 SuperPoD server, and Atlas 300I Duo inference card support this feature.
- Currently, only the Qwen3-32B, Qwen3-235B-A22B, Qwen3-30B-A3B, DeepSeek-R1, and DeepSeek-V3.1 models support this feature.
- To enable reasoning analysis for the DeepSeek-V3.1 model, include the following field in the request: `"chat_template_kwargs": {"enable_thinking": <bool>}`, or add `"enable_thinking": <bool>` to the `tokenizer_config.json` file.
- Currently, only the OpenAI inference API is supported.

## Description

[Table 1](#table1) describes the parameters required for enabling the thinking analysis feature.

**Table 1** `models` in `ModelConfig` <a id="table1"></a>

|Parameter|Value Type|Value Range|Configuration Description|
|--|--|--|--|
|enable_reasoning|bool|true<br>false|Specifies whether to enable model thinking analysis, separating the output into two fields: `reasoning_content` and `content`. `false`: disabled<br>`true`: enabled<br>Mandatory. The default value is `false`.|

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

2. Set serving parameters. Add the `enable_reasoning` field to the `config.json` file of the server according to [Table 1](#table1). For details about the serving parameters, see [Configuration Parameters (Serving)](../user_manual/service_parameter_configuration.md). The following is an example of parameter settings.

    The following uses Qwen3-32B as an example:

    ```json
     "ModelDeployConfig" :
            {
                "maxSeqLen" : 2560,
                "maxInputTokenLen" : 2048,
                "truncation" : 0,
                "ModelConfig" : [
                    {
                        "modelInstanceType" : "Standard",
                        "modelName" : "Qwen3-32B",
                        "modelWeightPath" : "/data/weight/Qwen3-32B",
                        "worldSize" : 1,
                        "cpuMemSize" : 0,
                        "npuMemSize" : -1,
                        "backendType" : "atb",
                        "trustRemoteCode" : false,
                        "async_scheduler_wait_time": 120,
                        "kv_trans_timeout": 10,
                        "kv_link_timeout": 1080,
                        "models": {
                                "qwen3": {"enable_reasoning": true}
                        }
                    }
                ]
            },
    ```

    > [!NOTE]NOTE
    >- Qwen3-30B-A3B: Change `qwen3` to `qwen3_moe`.
    >- DeepSeek-R1: Change `qwen3` to `deepseekv2` and change `model_type` in the DeepSeek-R1 weight file to `deepseek_v3`.
    >- DeepSeek-V3.2: Change `qwen3` to `deepseek_v32`.

3. Start the service.

   - **Installation using the `.whl` package:**

    ```bash
    mindie_llm_server
    ```

   - **Installation using the `.run` package:**

    ```bash
    ./bin/mindieservice_daemon
    ```

4. Send a request. For details about the parameters, see "Cluster Management Components" \> "Coordinator" \> "RESTful APIs" \> "User-Side APIs" \> "OpenAI Inference APIs" in *MindIE Motor Development Guide*.
