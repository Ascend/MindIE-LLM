# Multi-LoRA

Low-Rank Adaptation (LoRA) is an efficient parameter fine-tuning method. It decomposes the weight matrix of an LLM into the original weight matrix and a product of two low-rank matrices, that is, W' = W + BA. The number of parameters required for training matrix B and matrix A is far less than that required by the original weights. Therefore, the product result can be integrated into the linear layer and passed downwards, achieving lightweight fine-tuning of LLMs.

Multi-LoRA uses multiple LoRA weights for inference based on a base model. Each request carries a specified LoRA ID. During inference, the corresponding LoRA weight is dynamically matched. During service deployment, the LoRA weights and base model weights are pre-loaded to the graphics memory. A single inference request supports zero or at most one LoRA weight. For models with a large number of parameters, if the parameters cannot fit on a single device, tensor parallelism can be used.

The LoRA weight files must include the `adapter_config.json` and `adapter_model.safetensors` files. For details about the files, see [Table 1](#table1).

**Table 1** LoRA weight files <a id="table1"></a>

|File|Description|Example|
|--|--|--|
|adapter_config.json|Contains hyperparameters of the LoRA weights.|`r` (rank size in LoRA fine-tuning), `rank_pattern`, `lora_alpha` (scaling factor for the LoRA low-rank matrice), and `alpha_pattern`.|
|adapter_model.safetensors|Contains weights, which are saved as key-value pairs. The `base_model.model` prefix and the `lora_A.weight` and `lora_B.weight` suffixes are added to the start and end of the base model key name to form the LoRA weight key names.|Base model key name: `model.layers.9.self_attn.v_proj.weight` <br>LoRA weight key names: `base_model.model.model.layers.9.self_attn.v_proj.lora_A.weight` and `base_model.model.model.layers.9.self_attn.v_proj.lora_B.weight`|

## Constraints

- The Atlas 800I A2 inference server, Atlas 800I A3 SuperPoD server, and Atlas 300I Duo inference card support this feature.
- The number of LoRA weights is limited by the hardware memory. It is advised to keep the number ≤ 10.
- Dynamic loading and unloading of LoRA weights are supported only when ATB Models use Python to build graphs.
- LoRA weights can be carried by linear layers.
- This feature cannot be enabled together with the quantization, prefill-decode disaggregation, parallel decoding, SplitFuse, MTP, asynchronous scheduling, micro batch, or prefix cache features.
- This feature is supported only by Qwen2.5-7B, Qwen2.5-14B, Qwen2.5-32B, Qwen2.5-72B, Qwen3-32B, Llama 3.1 8B, Llama 3.1 70B, and Qwen2-72B.
- The length of the LoRA weight name cannot exceed 256 characters.
- This feature supports only vLLM, TGI, and vLLM-compatible OpenAI APIs.

## Parameter Description

[Table 2](#table2) describes the serving parameters required for enabling the Multi-LoRA feature.

**Table 2** Additional parameters for the Multi-LoRA feature: Parameters in `ModelDeployConfig` <a id="table2"></a>

|Parameter|Value Type|Value Range|Configuration Description|
|--|--|--|--|
|maxLoras|uint32_t|The upper limit is determined by the graphics memory and user requirements. The minimum value must be greater than 0.|Maximum number of LoRAs that can be loaded. This parameter is required when dynamic loading and unloading of LoRA weights are enabled. The default value is `0`.|
|maxLoraRank|uint32_t|The upper limit is determined by the graphics memory and user requirements. The minimum value must be greater than 0.|Maximum rank of the LoRA weights that can be loaded. This parameter is required when dynamic loading and unloading of LoRA weights are enabled. The default value is `0`.|
|**LoraModules**|-|-|-|
|name|string|The value can contain a maximum of 256 characters, including uppercase letters, lowercase letters, digits, hyphens (-), and underscores (_). It cannot start or end with a hyphen (-) or an underscore (_).|(Required) LoRA ID.|
|path|string|The maximum length of an absolute file path depends on the setting of the operating system (`PATH_MAX` in Linux). The minimum value is `1`.|(Required) Path of the LoRA weights. Security verification is performed on the path. The owner group and permission of the path must be the same as those of the execution user.|
|baseModelName|string|The value can contain a maximum of 256 characters, including uppercase letters, lowercase letters, digits, hyphens (-), periods (.), and underscores (_). It cannot start or end with a hyphen (-), period (.), or underscore (_).|(Required) Base model name. The value of this parameter must be the same as that of `modelName` in `ModelConfig`.|

## Inference

- Pure model usage:

    CANN and ATB Models have been installed in the environment. For details, see *MindIE Installation Guide*.

    > [!NOTE]NOTE
    > The following installation path is used as an example to
    > install ATB Models and initialize the ATB Models environment variables. The `${ATB_SPEED_HOME_PATH}` environment variable initialization is contained in the `set_env.sh` script of the model repository. Therefore, sourcing the `set_env.sh` script from the model repository also initializes the `${ATB_SPEED_HOME_PATH}` environment variable.

    The following uses Llama 3.1 70B as an example. After downloading the base model and LoRA weights, you can run the following command to perform a dialog test. Three requests form a batch for inference. The LoRA weight in each inference request is different. For details about the parameters in the `run_pa` script, see [Table 2](../user_manual/offline_inference.md#table2).

    Use `lora_modules` to specify the binding relationship between the base model and the LoRA weights.

    - The weight name is a weight alias, containing a maximum of 256 characters. It is used to specify the LoRA weight for inference in subsequent requests.
    - Multiple LoRA weights can be configured.

    ```bash
    cd ${ATB_SPEED_HOME_PATH}
    torchrun --nproc_per_node 8 --master_port 20030 -m examples.run_pa --model_path {Base_model_weight} --max_output_length 20 --max_batch_size 3 --input_dict '[{"prompt": "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?", "adapter": "{LoRA_weight_1_name}"}, {"prompt": "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?", "adapter": "{LoRA_weight_2_name}"}, {"prompt": "What is deep learning?", "adapter": "base"}]' --lora_modules '{"{LoRA_weight_1_name}": "{LoRA_weight_1_path}", "{LoRA_weight_2_name}": "{LoRA_weight_2_path}"}'
    ```

- Serving usage:

    > [!NOTE]NOTE
    > The `lora_adapter.json` file is no longer used for configuration. Add the `LoraModules` field to the `config.json` file of MindIE Motor to enable Multi-LoRA. The following describes the procedure.

    The following part uses Llama 3.1 70B as an example to describe how to use Multi-LoRA.

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

    2. Set serving parameters. Add the `maxLoras`, `maxLoraRank`, and `LoraModules` fields to the `config.json` file of the server. For details about the parameter fields, see [Table 2](#table2). For details about the serving parameters, see [Configuration Parameters (Serving)](../user_manual/service_parameter_configuration.md). The following is an example of parameter configuration.

        ```json
        {
            "ServerConfig" :
            {
                "ipAddress" : "127.0.0.1",
                "managementIpAddress" : "127.0.0.2",
                "port" : 1025,
                "managementPort" : 1026,
            },
            "BackendConfig": {
                "backendName" : "mindieservice_llm_engine",
                "modelInstanceNumber" : 1,
                "npuDeviceIds" : [[0,1,2,3,4,5,6,7]],
                "tokenizerProcessNumber" : 8,
                "multiNodesInferEnabled": false,
                "multiNodesInferPort": 1120,
                "interNodeTLSEnabled": true,
                "interNodeTlsCaPath": "security/grpc/ca/",
                "interNodeTlsCaFiles": ["ca.pem"],
                "interNodeTlsCert": "security/grpc/certs/server.pem",
                "interNodeTlsPk": "security/grpc/keys/server.key.pem",
                "interNodeTlsCrlPath" : "security/grpc/certs/",
                "interNodeTlsCrlfiles" : ["server_crl.pem"],
                "ModelDeployConfig":
                {
                    "maxSeqLen" : 2560,
                    "maxInputTokenLen" : 2048,
                    "truncation" : 0,
                    "ModelConfig" : [
                        {
                            "modelInstanceType": "Standard",
                            "modelName" : "llama3.1-70b",
                            "modelWeightPath" : "/data/weights/llama3.1-70b-safetensors",
                            "worldSize" : 8,
                            "cpuMemSize" : 5,
                            "npuMemSize" : -1,
                            "backendType": "atb",
                            "trustRemoteCode": false
                        }
                    ],
                    "maxLoras": 4,
                    "maxLoraRank": 296,
                    "LoraModules" :[{
                    "name" : "adapter1",
                    "path" : "/data/lora_model_weights/llama3.1-70b-lora",
                    "baseModelName" : "llama3.1-70b"
                    }]
                },
            }
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

    4. Dynamically load, unload, or query LoRA.

        Loading request:

        ```bash
        curl -X POST http://127.0.0.2:1026/v1/load_lora_adapter \
        -H "Content-Type: application/json" \
        -d '{
            "lora_name": "adapter2",
            "lora_path": "/data/lora_model_weights/llama3.1-70b-lora"
        }'
        ```

        Unloading request:

        ```bash
        curl -X POST 127.0.0.2:1026/v1/unload_lora_adapter -d '{
            "lora_name": "adapter2"
        }'
        ```

        Querying request:

        ```bash
        curl http://127.0.0.1:1025/v1/models
        ```

    5. Send a request.

        `model` can be set to the base model name (value of `modelName` under the `ModelConfig` field in the `config.json` file) or the LoRA ID (value of `name` under the `LoraModules` field in the `config.json` file). If `model` is set to the base model name, no LoRA weights are used for inference. If `model` is set to the LoRA ID, the base model weights and the specified LoRA weights are used for inference.

        ```bash
        curl https://127.0.0.1:1025/generate \
        -H "Content-Type: application/json" \
        --cacert ca.pem --cert client.pem  --key client.key.pem \
        -X POST \
        -d '{
        "model": "${Base model name}",
        "prompt": "Taxation in Puerto Rico -- The Commonwealth government has its own tax laws and Puerto Ricans are also required to pay some US federal taxes, although most residents do not have to pay the federal personal income tax. In 2009, Puerto Rico paid $3.742 billion into the US Treasury. Residents of Puerto Rico pay into Social Security, and are thus eligible for Social Security benefits upon retirement. However, they are excluded from the Supplemental Security Income.\nQuestion: is federal income tax the same as social security?\nAnswer:",
        "max_tokens": 20,
        "temperature": 0
        }'

        curl https://127.0.0.1:1025/generate \
        -H "Content-Type: application/json" \
        --cacert ca.pem --cert client.pem  --key client.key.pem \
        -X POST \
        -d '{
        "model": "adapter1",
        "prompt": "Taxation in Puerto Rico -- The Commonwealth government has its own tax laws and Puerto Ricans are also required to pay some US federal taxes, although most residents do not have to pay the federal personal income tax. In 2009, Puerto Rico paid $3.742 billion into the US Treasury. Residents of Puerto Rico pay into Social Security, and are thus eligible for Social Security benefits upon retirement. However, they are excluded from the Supplemental Security Income.\nQuestion: is federal income tax the same as social security?\nAnswer:",
        "max_tokens": 20,
        "temperature": 0
        }'
        ```
