# KV Cache池化

## 特性介绍

在当前大语言模型的推理系统中，KV Cache是广泛采用的缓存机制。MindIE在此基础上进一步引入Prefix Cache技术，能够在请求命中缓存时显著减少Prefill阶段的计算耗时，并有效节省显存占用。

然而，Prefix Cache默认仅使用片上内存，其容量有限，难以缓存大量前缀信息。为此，KV Cache池化特性实现了存储层级的扩展——支持将DRAM甚至SSD等更大容量的存储介质纳入前缀缓存池，从而突破片上内存的容量限制。该机制有效提升了Prefix Cache的命中率，显著降低大模型推理的成本。

## 限制与约束

-  Atlas 800I A2 推理服务器和Atlas 300I Duo 推理卡支持此特性。
-  其他限制与约束和Prefix Cache 相同，请参考[限制与约束](prefix_cache.md#限制与约束)。
-  当前仅支持DRAM池化，即和Prefix Cache叠加后两级缓存。
-  使用KV Cache池化特性时，必须同时打开Prefix Cache特性。
-  底层采用基于HCCL单边通信的池化后端，会额外占用片上内存，主要包括HCCL建链所需的队列显存。具体说明如下：
   -  每条HCCL链路占用4MB显存。同时受HCCL底层能力限制，最大建链数为512条。
   -  根据构建池化的总卡数，额外显存占用的计算公式为：（参与池化节点的总卡数/总die数-1）*4MB。
   -  系统支持通过下调显存因子释放空间，用于HCCL建链。显存因子每下调0.01，可释放600MB显存。显存因子最大可下调0.01，以满足当前池化建链数上限。同时，显存因子下调后，支持的上下文长度会相应降低。
   -  会扩容场景：建议按最大显存因子下调值0.04预留显存，避免有新节点动态加入时因HCCL建链导致OOM。例如：默认显存因子是0.92，扩容场景应设置为0.08。
   -  不会扩容的场景：根据实际参与池化的节点总卡数，按公式计算HCCL建链所需显存，结合计算结果，决定显存因子下调幅度。例如：以Atlas 800I A3服务器，4机+4机场景为例，HCCL建链额外显存占用为：（8*16-1）*4MB=508MB。默认显存因子为0.92，下调0.01即可满足需求（释放约600MB > 508MB），并注意相应降低上下文长度。
- 底层采用基于HCCL单边通信的池化后端。受HCCL底层能力限制，单次HCCL建链数量上限为512条。因此，在构建统一逻辑池时，建议总卡/die数量≤512，以保证在长时间稳定传输过程中，不会因频繁断链、重建链而导致性能下降。

## 参数说明

开启KV Cache池化特性需要配置的补充参数如[表1](#table1)所示。

**表 1**  KV Cache池化特性补充参数：**BackendConfig中的参数**  <a id="table1"></a>

|配置项|取值类型|取值范围|配置说明|
|--|--|--|--|
|kvPoolConfig|std::string|{"backend"："*kv_pool_backend_name*", <br>"configPath":"*/path/to/your/config/file*"，<br>"asyncWrite":false}|<li>backend为指定的KV Cache池化后端。<ul><li>设置为“”，表示关闭KV Cache池化。</li><li>设置为对应池化后端的名称，表示开启KV Cache池化。</li></ul></li><li>configPath为传入池化后端所需的配置文件路径。</li><li>asyncWrite为池化KV Cache异步写开关。<ul><li>不设置或设置为false，表示关闭KV Cache的异步写。</li><li>设置为true，表示开启KV Cache的异步写。</li></ul></li>|


## 执行推理

1. 打开Server的config.json文件。

    ```bash
    cd {MindIE安装目录}/latest/mindie-service/
    vi conf/config.json
    ```

2. 配置服务化参数。使用KV Cache池化特性时，必须打开Prefix Cache特性。

    请根据[表1](prefix_cache.md#table1)\~[表3](prefix_cache.md#table3)，和[表1](#table1)，在Server的config.json文件添加相应参数。其他服务化参数说明请参见[配置参数说明（服务化）](../user_manual/service_parameter_configuration.md)

    下面以DeepSeek-R1模型, 开启Prefix Cache+KV Cache池化为例：

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
            "kvPoolConfig" : {"backend"："kv_pool_backend_name", "configPath":"/path/to/your/config/file"},
        "ModelDeployConfig" :
            {
                "maxSeqLen" : 20000,
                "maxInputTokenLen" : 4096,
                "truncation" : 0,
                "ModelConfig" : [
                    {
                        "modelInstanceType" : "Standard",
                        "modelName" : "dsr1",
                        "modelWeightPath" : "/*权重路径*/deepseek_r1_w8a8_mtp",
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

3. 拉起池化后端对应的中心化服务Master Service，具体安装和拉起命令，请参考[KV Cache池化使用指导](../../../../mindie_llm/text_generator/mempool/README.md)。
4. 启动服务。

    ```bash
    ./bin/mindieservice_daemon
    ```

5. 第一次使用以下指令发送请求，prompt为第一轮问题。如需使用到Prefix Cache/池化特性，第二次请求的prompt需要与第一次的prompt有一定长度的公共前缀，常见使用场景有多轮对话和few-shot学习等。具体curl命令请参考[Prefix Cache章节发送请求](prefix_cache.md)的命令和内容。
6. 发送后续请求，由于片上内存命中率优先级高于DRAM池化，如果需要真实从池子命中，需要保证片上内存中无法命中。

