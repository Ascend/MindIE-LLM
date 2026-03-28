# Prefix Cache

当前大语言模型推理系统普遍采用KV Cache缓存机制，但该机制存在以下两个问题：

1. 随着LLM支持的序列长度不断增长，KV Cache所需要的显存资源也急剧增加。
2. KV Cache只对当前session有效，如果跨session存在重复Token序列的情况下无法实现复用。

Prefix Cache通过哈希表保留session结束后的KV Cache，新的session请求在哈希表中查找是否存在相同的Token序列，即可复用之前计算好的KV Cache，从而实现跨session的KV Cache复用。

其优势主要包括：

- **更短的prefill时间**：由于跨session的重复Token序列对应的KV Cache可以复用，那么就可以减少一部分前缀Token的KV Cache计算时间，从而减少prefill的时间。
- **更高效的显存使用**：当正在处理的sessions相互之间存在公共前缀时，公共前缀部分的KV Cache可以共用，不必重复占用多份显存。

## 限制与约束<a name="section204315644813"></a>

- Atlas 800I A2 推理服务器和Atlas 300I Duo 推理卡和和Atlas 800I A3 超节点服务器支持此特性。
- Qwen2系列、Qwen2.5系列、Qwen3系列、DeepSeek-R1和DeepSeek-V3/V3.1模型支持对接此特性。
- 当跨session公共前缀Token数大于等于block size时，才会进行公共前缀Token的KV Cache复用。
- Prefix Cache支持的量化特性：W4A8量化、W8A8量化、PDMIX量化与稀疏量化，其他量化特性暂不支持。
- 该特性不能和Multi-LoRA特性同时使用。
- 该特性可以和PD分离、并行解码、MTP、kvcache池化、异步调度、SplitFuse特性、context parallel + sequence parallel、C8量化同时使用。
- 该特性支持n、best\_of、use\_beam\_search后处理参数。
- PD分离场景下，仅P节点需要开启该特性。
- 前缀复用率低或者没有复用的情况下，不建议开启该特性。
- 不支持prefix cache + context parallel + sequence parallel + function call(multiturn)的叠加

## 参数说明

开启Prefix Cache特性需要配置的补充参数如[表1](#table1)~[表3](#table3)所示。

**表 1**  Prefix Cache特性补充参数1：**ModelDeployConfig中的ModelConfig参数**  <a id="table1"></a>

|配置项|取值类型|取值范围|配置说明|
|--|--|--|--|
|plugin_params|std::string|"{\"plugin_type\":\"prefix_cache\"}"|<ul><li>设置为"{\"plugin_type\":\"prefix_cache\"}"，表示执行Prefix Cache。</li><li>不需要生效任何插件功能时，请删除该配置项字段。</li></ul>|

**表 2**  Prefix Cache特性补充参数2：**ScheduleConfig的参数**  <a id="table2"></a>

|配置项|取值类型|取值范围|配置说明|
|--|--|--|--|
|enablePrefixCache|-|-|该字段已无需配置，目前版本按老版本方式配置无影响。<br>该字段预计下线时间：2026年Q1版本。|

**表 3**  Prefix Cache特性补充参数3：**ModelConfig中的models参数**  <a id="table3"></a>

|配置项|取值类型|取值范围|配置说明|
|--|--|--|--|
|deepseekv2|
|kv_cache_option|
|enable_nz|bool|<ul><li>true</li><li>false</li></ul>|是否开启KV Cache NZ格式。<br><ul><li>仅DeepSeek-R1、DeepSeek-V3和DeepSeek-V3.1模型支持此特性。FA3量化场景下自动使能NZ格式。</li><li>DeepSeek-R1、DeepSeek-V3和DeepSeek-V3.1模型必须开启此开关，其余模型关闭。</li><li>默认值：false</li></ul>|

## 执行推理

以多轮对话为例，简单介绍Prefix Cache如何使用。

1. 打开Server的config.json文件。

    ```bash
    cd {MindIE安装目录}/latest/mindie-service/
    vi conf/config.json
    ```

2. 配置服务化参数。在Server的config.json文件中按照[表1](#table1)\~[表3](#table3)添加相应参数，服务化参数说明请参见[配置参数说明（服务化）](../user_manual/service_parameter_configuration.md)章节，参数配置示例如下。

    下面以DeepSeek-R1模型，只开启Prefix Cache特性为例。

    ```json
    "ModelDeployConfig" :
    {
       "maxSeqLen" : 2560,
       "maxInputTokenLen" : 2048,
       "truncation" : 0,
       "ModelConfig" : [
         {
             "plugin_params": "{\"plugin_type\":\"prefix_cache\"}",
             "modelInstanceType" : "Standard",
             "modelName" : "DeepSeek-R1_w8a8",
             "modelWeightPath" : "/data/weights/DeepSeek-R1_w8a8",
             "worldSize" : 8,
             "cpuMemSize" : 5,
             "npuMemSize" : -1,
             "backendType" : "atb",
             "trustRemoteCode" : false,
             "models": {
                 "deepseekv2": {
                     "kv_cache_options": {"enable_nz": true}
                  }
             }
          }
       ]
    },
    ```

    > [!NOTE]说明 
    >- 如果是非DeepSeek模型，不需要配置“models”字段。
    >- 如果需要特性叠加使用，如：Prefix Cache和MTP叠加，需使用英文逗号将特性名称隔开。方法如下：
    > **"plugin\_params": "\{\\"plugin\_type\\":\\"mtp,prefix\_cache\\",\\"num\_speculative\_tokens\\": 1\}"**,

3. 启动服务。

    ```bash
    ./bin/mindieservice_daemon
    ```

4. 第一次使用以下指令发送请求，prompt为第一轮问题。

    如需使用到Prefix Cache特性，第二次请求的prompt需要与第一次的prompt有一定长度的公共前缀，常见使用场景有多轮对话和few-shot学习等。

    ```bash
    curl https://127.0.0.1:1025/generate \
    -H "Content-Type: application/json" \
    --cacert ca.pem --cert client.pem  --key client.key.pem \
    -X POST \
    -d '{
    "inputs": "Question: Parents have complained to the principal about bullying during recess. The principal wants to quickly resolve this, instructing recess aides to be vigilant. Which situation should the aides report to the principal?\na) An unengaged girl is sitting alone on a bench, engrossed in a book and showing no interaction with her peers.\nb) Two boys engaged in a one-on-one basketball game are involved in a heated argument regarding the last scored basket.\nc) A group of four girls has surrounded another girl and appears to have taken possession of her backpack.\nd) Three boys are huddled over a handheld video game, which is against the rules and not permitted on school grounds.\nAnswer:",
    "parameters": {"max_new_tokens":512}
    }'
    ```

5. 第二次发送请求，prompt为：第一轮问题+第一轮答案+第二轮问题，此时第一轮问题为可复用的公共前缀（实际复用部分可能不是第一轮问题的完整prompt；由于cache实现以block为单位，Prefix Cache以blocksize的倍数储存，如第一轮问题prompt的token数量为164，当blocksize为128时，实际复用部分只有前128token）。

    ```bash
    curl https://127.0.0.1:1025/generate \
    -H "Content-Type: application/json" \
    --cacert ca.pem --cert client.pem  --key client.key.pem \
    -X POST \
    -d '{
    "inputs": "Question: Parents have complained to the principal about bullying during recess. The principal wants to quickly resolve this, instructing recess aides to be vigilant. Which situation should the aides report to the principal?\na) An unengaged girl is sitting alone on a bench, engrossed in a book and showing no interaction with her peers.\nb) Two boys engaged in a one-on-one basketball game are involved in a heated argument regarding the last scored basket.\nc) A group of four girls has surrounded another girl and appears to have taken possession of her backpack.\nd) Three boys are huddled over a handheld video game, which is against the rules and not permitted on school grounds.\nAnswer:c) A group of four girls has surrounded another girl and appears to have taken possession of her backpack.\nExplanation: The principal wants to quickly resolve this, instructing recess aides to be vigilant. The principal is concerned about bullying during recess. The principal wants the aides to report any bullying behavior to him. The principal is not concerned about the other situations.\nQuestion: If the aides confront the group of girls from situation (c) and they deny bullying, stating that they were merely playing a game, what specific evidence should the aides look for to determine if this is a likely truth or a cover-up for bullying?\nAnswer:",
    "parameters": {"max_new_tokens":512}
    }'
    ```
