# 并行解码

在LLM的推理场景中，传统的Auto-Regressive Decoding慢，是因为step-by-step导致了并发性不够。推理阶段属于内存带宽受限而计算资源过剩的阶段。因此，并行解码特性就是采用处理器中常用的“Speculative Execution”优化技术，通过额外的计算资源完成推测执行，提升并发性。但是，由于开启并行解码会使用Prompt输入维护前缀树和草稿token map，所以会对首token时延有一定影响。

并行解码的优势：

针对足够长度的输入输出或代码生成等场景的小batch推理，并行解码特性可利用算力优势弥补访存带宽受限的影响，提升算力利用率。同时因为通过验证token的比率会直接影响到并行解码的收益，因此贪婪场景更能充分发挥并行解码的效果，而采样或惩罚类操作会影响并行解码的收益空间。

为了发挥并行解码的优势，需满足如下前提：

1.  当前的并发数不高，属于内存带宽受限、计算资源有冗余的情况。
2.  有较长的输入作为猜测token的初步来源。
3.  并行解码主要通过减少推理步数获取增益，因此需要一定长度的输出才有性能提升效果。

目前支持两种并行解码算法，差异主要在于候选token生成的方式不同。如[表1](#table1)所示。

**表 1**  并行解码算法  <a id="table1"></a>

|并行解码算法|候选token生成方式|适用场景|
|--|--|--|
|memory_decoding|利用trie tree（前缀树）缓存模型历史的输入输出，从中获取候选token。|代码生成或检索类场景。|
|lookahead|基于Jacobi迭代并辅以Prompt以及输出结果生成候选token。|文本生成、对话系统及多样化查询回答。|


## 限制与约束

-  Atlas 800I A2 推理服务器和Atlas 300I Duo 推理卡支持此特性。
-  LLaMA3系列、Qwen2系列、Qwen2.5系列、Qwen3-14B和Qwen3-32B模型支持对接此特性。
-  并行解码支持的量化特性：W8A8量化与稀疏量化，其他量化特性暂不支持。
-  该特性不能和PD分离、Multi-LoRA、SplitFuse、长序列、MTP、异步调度以及多机推理特性同时使用。
-  该特性暂不支持n、best\_of、use\_beam\_search、logprobs、top\_logprobs等与多序列推理相关的后处理参数。
-  并行解码场景暂不支持流式推理。
-  并行解码惩罚类后处理仅支持重复惩罚。
-  并行解码场景暂不支持开启健康检查HealthCheck。
-  lookahead和memory\_decoding算法不可同时使能。

## 参数说明

开启并行解码特性，需要配置的参数如[表2](#table2)~[表6](#table6)所示：

**表 2**  memory\_decoding补充参数1：**ModelDeployConfig中的ModelConfig参数**  <a id="table2"></a>

|配置项|取值类型|取值范围|配置说明|
|--|--|--|--|
|plugin_params|std::string|plugin_type：memory_decoding<br>decoding_length：[1, 16]<br>dynamic_algo：true或false|<ul><li>plugin_type配置memory_decoding，表示当前选择memory_decoding并行解码。</li><li>decoding_length为memory_decoding算法中的参数，表示候选token的最大长度，默认值16。</li><li>dynamic_algo为可选参数，配为true时表示开启动态自适应候选长度功能，默认值False。</li><li>不需要生效任何插件功能时，请删除该配置项字段。</li><li>配置示例：{\"plugin_type\":\"memory_decoding\",\"decoding_length\": 16,\"dynamic_algo\": true}或{\"plugin_type\":\"memory_decoding\",\"decoding_length\": 16}</li></ul>|


**表 3**  memory\_decoding补充参数2：**ModelDeployConfig的参数**

|配置项|取值类型|取值范围|配置说明|
|--|--|--|--|
|speculationGamma|uint32_t|与plugin参数配置有关|memory_decoding时，该值配置应大于等于decoding_length。<br>建议值：等于decoding_length。|


**表 4**  memory\_decoding补充参数3：**ScheduleConfig的参数**

|配置项|取值类型|取值范围|配置说明|
|--|--|--|--|
|maxIterTimes|uint32_t|与plugin参数配置有关|如果dynamic_algo为true，该值需大于等于期望输出的长度+speculationGamma的值。<br>例：期望最大输出长度为512，则该值需要配置>=512+speculationGamma。|


**表 5**  lookahead补充参数1：**ModelDeployConfig中的ModelConfig参数**

|配置项|取值类型|取值范围|配置说明|
|--|--|--|--|
|plugin_params|std::string|plugin_type：la<br>level ：[3, 16]<br>window ：[1, 16]<br>guess_set_size ：[1, 16]|plugin_type配置la，表示当前选择lookahead并行解码。<br>level/window/guess_set_size为lookahead算法中的N/W/G参数，默认值为4/5/5，且每个参数可配置的上限不超过16。配置示例："{\"plugin_type\":\"la\",\"level\": 4,\"window\": 5,\"guess_set_size\": 5}"|


**表 6**  lookahead补充参数2：**ModelDeployConfig的参数**  <a id="table6"></a>

|配置项|取值类型|取值范围|配置说明|
|--|--|--|--|
|speculationGamma|uint32_t|与plugin参数配置有关|lookahead中，配置值应大于等于(N-1)*(W+G)<br>建议值：等于(N-1)*(W+G)。|


## 执行推理<a name="section1788515529541"></a>

1. 打开Server的config.json文件。

    ```
    cd {MindIE安装目录}/latest/mindie-service/
    vi conf/config.json
    ```

2. 配置服务化参数。在Server的config.json文件中按照[表2](#table1)~[表6](#table6)添加相应参数，服务化参数说明请参见[配置参数说明（服务化）](../user_manual/service_parameter_configuration.md)章节，参数配置示例如下。

    memory\_decoding算法的并行解码配置样例：

    ```json
    "ModelDeployConfig" :
    {
        "maxSeqLen" : 2560,
        "maxInputTokenLen" : 2048,
        "truncation" : false,
        "speculationGamma": 16,
        "ModelConfig" : [
            {
                "plugin_params":"{\"plugin_type\":\"memory_decoding\",\"decoding_length\":16,\"dynamic_algo\":true}",
                "modelInstanceType" : "Standard",
                "modelName" : "llama3-70b",
                "modelWeightPath" : "/data/weights/llama3-70b",
                "worldSize" : 4,
                "cpuMemSize" : 5,
                "npuMemSize" : -1,
                "backendType" : "atb",
                "trustRemoteCode" : false
            }
        ]
    }
    ```

    lookahead算法的并行解码配置样例：

    ```json
    "ModelDeployConfig" :
    {
        "maxSeqLen" : 2560,
        "maxInputTokenLen" : 2048,
        "truncation" : false,
        "speculationGamma": 30,
        "ModelConfig" : [
            {
                "plugin_params":"{\"plugin_type\":\"la\",\"level\":4,\"window\":5,\"guess_set_size\":5}",
                "modelInstanceType" : "Standard",
                "modelName" : "Qwen2.5-7B-Instruct",
                "modelWeightPath" : "/data/weights/Qwen2.5-7B-Instruct",
                "worldSize" : 1,
                "cpuMemSize" : 5,
                "npuMemSize" : -1,
                "backendType" : "atb",
                "trustRemoteCode" : false
            }
        ]
    }
    ```

3. 启动服务。

    ```bash
    ./bin/mindieservice_daemon
    ```

