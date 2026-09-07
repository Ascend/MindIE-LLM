# Thinking Budget

Some large models include their reasoning process in the output. This feature limits the model's thinking depth: when the reasoning exceeds the `thinking_budget`, the system truncates the chain of thought using a prompt, encouraging the model to stop reasoning early. This is useful for scenarios requiring a flexible trade-off between response speed and answer quality.

## Constraints

- The Atlas 800I A2 inference server, Atlas 800I A3 SuperPoD server, and Atlas 300I Duo inference card support this feature.

- Currently, only the Qwen3-32B, Qwen3-235B-A22B, and Qwen3-30B-A3B models support this feature.

- To enable the thinking budget, include the following field in your request:  
`"chat_template_kwargs": {"thinking_budget": <uint32_t>}`.  
The value must be in the range `[1, MAX_UINT32_T]`.

- Currently, only the OpenAI inference interfaces are supported.

- This feature currently does not support being enabled simultaneously with postprocessing parameters related to multi-sequence inference, such as `use_beam_search`.

## Parameter Description

To enable the Thinking Budget feature, the parameters that need to be configured are shown in [Table 1](#table1).

**Table 1** Thinking Budget parameter: `models` in `ModelConfig` <a id="table1"></a>

|Configuration Item|Value Type|Length Range|Configuration Description|
|--|--|--|--|
|early_stopping_text|string|[1,1024]|End-of-thinking prompt.<br>Used to truncate model output when the thinking budget is exceeded after enabling `thinking_budget`. Prompt formats vary by model. For Qwen-series models, refer to the parameter configuration example below.|

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

2. Configure the serving parameters. Add the `early_stopping_text` field to the Server's `config.json` file according to [Table 1](#table1). For details on serving parameters, see [Parameter Configuration (Serving)](../user_manual/service_parameter_configuration.md). A parameter configuration example is shown below.

    Qwen3-32B is used as an example:

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
                                "qwen3": {"early_stopping_text": "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"}
                        }
                    }
                ]
            },
    ```

    > [!NOTE]
    >- Qwen3-30B-A3B model: The `qwen3` field should be modified to `qwen3_moe`.
    >- When `thinking_budget` is set too low, the model may fall back to outputting in the same language as the prompt.

3. Start the service. For PD colocation scenarios, refer to "Quick Start" \> "[Starting the Service](https://www.hiascend.com/document/detail/en/mindie/310/mindiellm/llmdev/mindie_motor_cpp/user_guide/quick_start.md)" in *MindIE Motor CPP Developer Guide*.
For PD disaggregation scenarios, refer to "Cluster Service Deployment" > "[PD Disaggregation](https://www.hiascend.com/document/detail/en/mindie/310/mindiellm/llmdev/mindie_motor_cpp/user_guide/service_deployment/pd_separation_service_deployment.md)" in *MindIE Motor CPP Developer Guide*.  
4. Send a request. For parameter descriptions, see the "[Serving API Usage Guide](../user_manual/service_APIs_usage_guidance.md)" section in *MindIE LLM Developer Guide*
