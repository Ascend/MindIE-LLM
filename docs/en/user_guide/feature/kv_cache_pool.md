# KV Cache Pooling

## Feature Introduction

Currently, the KV cache mechanism is widely used in LLM inference systems. MindIE further introduces the prefix cache technology to significantly reduce the computation time in the prefill phase and effectively save the graphics memory when requests hit the cache.

However, the prefix cache uses only on-chip memory by default, which has limited capacity and cannot cache a large amount of prefix information. To address this issue, the KV cache pooling feature is used to extend the storage hierarchy, which allows larger-capacity storage media such as DRAM and SSDs to be added to the prefix cache pool, thereby breaking the capacity limit of the on-chip memory. This mechanism effectively improves the prefix cache hit ratio and significantly reduces the cost of LLM inference.

## Constraints

- The Atlas 800I A2 inference server and Atlas 300I Duo inference card support this feature.
- Other constraints are the same as those of the prefix cache. For details, see [constraints](./prefix_cache.md#constraints).
- Currently, only DRAM pooling is supported, forming a two-level cache structure with the prefix cache.
- To use the KV cache pooling feature, the prefix cache feature must be enabled.
- The underlying layer uses the pooling backend based on HCCL one-sided communication, which occupies additional on-chip memory, mainly including the queue memory required for HCCL link setup. The details are as follows:
   - Each HCCL link occupies 4 MB of graphics memory. Due to limitations of the HCCL backend, a maximum of 512 links can be established.
   - Based on the number of cards participating in pooling, the additional graphics memory usage is calculated as follows: (Total pooled cards/total Number of dies – 1) × 4 MB.
   - The system supports releasing space by reducing the graphics memory factor for HCCL link setup. Each time the graphics memory factor is reduced by 0.01, 600 MB of graphics memory can be released. The graphics memory factor can be reduced by up to 0.01, meeting the link setup upper limit requirements in the pooling scenario. Note that reducing the graphics memory factor also decreases the supported context length.
   - In scale-out scenarios, you are advised to reserve graphics memory by reducing the graphics memory factor by 0.04. This prevents OOM caused by HCCL link setup when new nodes are dynamically added. For example, if the default graphics memory factor is 0.92, set it to 0.08 in scale-out scenarios.
   - In non-scale-out scenarios, calculate the graphics memory required for HCCL link setup based on the total number of cards participating in pooling across all nodes and reduce the graphics memory factor accordingly. For example, when 4 + 4 Atlas 800I A3 servers are configured, the additional graphics memory required for HCCL link setup is calculated as follows: (8 × 16 – 1) × 4 MB = 508 MB. The default graphics memory factor is 0.92. Reducing it by 0.01 can meet the requirements (about 600 MB of graphics memory is released, which is greater than 508 MB). Note that the context length will be reduced accordingly.
- The underlying layer uses the pooling backend based on HCCL one-sided communication. Due to limitations of the HCCL backend, a maximum of 512 HCCL links can be established at a time. Therefore, when a unified logical pool is built, it is recommended that the total number of cards or dies be equal to or less than 512, so that performance remains stable during long-duration transmissions and is not impacted by frequent link disconnections or link rebuilding.

## Parameter Description

[Table 1](#table1) describes the parameters required for enabling the KV cache pooling feature.

**Table 1** KV cache pooling parameter in `BackendConfig` <a id="table1"></a>

|Parameter|Value Type|Value Range|Configuration Description|
|--|--|--|--|
|kvPoolConfig|std::string|{"backend":"*kv_pool_backend_name*", <br>"configPath":"*/path/to/your/config/file*",<br>"asyncWrite":false}|<li>`backend`: backend of KV cache pooling. <ul><li>`""`: KV cache pooling disabled. </li><li>Backend name: KV cache pooling enabled. </li></ul></li><li>`configPath`: path to the configuration file required by the pooling backend. </li><li>`asyncWrite`: asynchronous write switch of the pooled KV cache. <ul><li>If this parameter is not set or set to `false`, asynchronous KV Cache writes are disabled. </li><li>If this parameter is set to `true`, asynchronous KV Cache writes are enabled.</li></ul></li>|

## Inference

1. Open the `config.json` file of the server.

    ```bash
    cd {MindIE_installation_directory}/mindie_llm/
    vi conf/config.json
    ```

2. Set serving parameters. To use the KV cache pooling feature, the prefix cache feature must be enabled.

    Add the corresponding parameters to the `config.json` file of the server based on [Table 1](prefix_cache.md#table1) to [Table 3](prefix_cache.md#table3) and [Table 1](#table1). For details about other serving parameters, see [Configuration Parameter Description (serving)] (../user_manual/service_parameter_configuration.md).

    The following uses the DeepSeek-R1 model as an example to describe how to enable the prefix cache and KV cache pooling features:

    ```json
    "BackendConfig" : {
            "backendName" : "mindieservice_llm_engine",
            "modelInstanceNumber" : 1,
            "npuDeviceIds" : [[0,1,2,3,4,5,6,7]],
            "tokenizerProcessNumber" : 8,
            "multiNodesInferEnabled" : true,
            "multiNodesInferPort" : 1120,
            "interNodeTLSEnabled" : false,
            "interNodeTlsCaPath" : "security/grpc/ca/",
            "interNodeTlsCaFiles" : ["ca.pem"],
            "interNodeTlsCert" : "security/grpc/certs/server.pem",
            "interNodeTlsPk" : "security/grpc/keys/server.key.pem",
            "interNodeTlsCrlPath" : "security/grpc/certs/",
            "interNodeTlsCrlFiles" : ["server_crl.pem"],
            "kvPoolConfig" : {"backend":"kv_pool_backend_name", "configPath":"/path/to/your/config/file"},
        "ModelDeployConfig" :
            {
                "maxSeqLen" : 20000,
                "maxInputTokenLen" : 4096,
                "truncation" : 0,
                "ModelConfig" : [
                    {
                        "modelInstanceType" : "Standard",
                        "modelName" : "dsr1",
                        "modelWeightPath": "/*Weight_path*/deepseek_r1_w8a8_mtp,"
                        "worldSize" : 8,
                        "cpuMemSize" : 0,
                        "npuMemSize" : -1,
                        "backendType" : "atb",
                        "trustRemoteCode" : false,
                        "async_scheduler_wait_time": 120,
                        "kv_trans_timeout": 10,
                        "kv_link_timeout": 1080,
                        "dp": 2,
                        "sp": 1,
                        "tp": 8,
                        "moe_ep": 4,
                        "moe_tp": 4,
                        "plugin_params": "{\"plugin_type\":\"prefix_cache\"}",
                        "models": {
                            "deepseekv2": {
                                "enable_mlapo_prefetch": true,
                                "kv_cache_options": {
                                "enable_nz": true
                                }
                           }
                       }
                    }
                ]
            },
    ```

3. Start the centralized master service corresponding to the pooling backend. For details about the installation and startup commands, see [KV Cache Pooling Usage Guide](mempool.md).
4. Start the service.

    ```bash
    mindie_llm_server
    ```

5. Send a request for the first time. The prompt is the first round of questions. To use prefix cache/KV cache pooling, the prompt of the second request must have a common prefix with the prompt of the first request. Common application scenarios include multi-round dialog and few-shot learning. For details about the `curl` command, see [sending a request in Prefix Cache](prefix_cache.md).
6. Send subsequent requests. The on-chip memory has higher cache-hit priority than the DRAM pool. Requests can hit the DRAM pool only when on-chip cache misses occur.
