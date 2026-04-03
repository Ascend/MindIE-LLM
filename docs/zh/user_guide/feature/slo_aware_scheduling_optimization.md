# SLO调度优化

SLO（Service Level Objective，服务级别目标）指在设定时间段内为特定指标设定的目标值。为应对客户端的高并发请求，在确保SLO的前提下提升系统吞吐量，目前提供以下两种实现手段：

1、基于TTFT/TPOT时延预测和LLF（Least Laxity First，松弛度优先）算法的PD阶段选择算法

该算法通过采集当前的TTFT和TPOT时延数据进行拟合建模，预测每次Prefill和Decode阶段的处理时间，并使用松弛度优先（Least Laxity First，LLF）算法决定下一批Batch是执行Prefill还是Decode。该算法适用于对TTFT和TPOT均有严格要求的场景，能够在高并发环境下，满足SLO的前提下提升吞吐量。

2、基于实时TPOT感知的动态BatchSize调整算法

该算法持续监测系统当前的TPOT时延，并与SLO中设定的Decode时延目标进行比对，根据比对结果，动态调整maxPrefillBatchSize和maxBatchSize，避免所有请求都进入片上内存导致系统拥塞影响吞吐。该算法适用于对TPOT有强要求的场景，能够在高并发环境下，优先保障已进入片上内存请求的响应。由于TPOT采集存在实时波动，最终的实时时延与配置目标之间可能存在约10%的偏差。

## 限制与约束

-  仅Atlas 800I A2 推理服务器支持此特性。
-  DeepSeek-R1、DeepSeek-V3、Qwen系列模型支持对接此特性。
-  仅适用于PD混部场景，无法与splitfuse特性同时打开。
-  此特性的收益场景主要在短输出（256以下）场景，随着输出长度变长，吞吐收益会下降。

## 参数说明

开启SLO调度优化特性，需要配置的参数如*[表1](#table1)所示。

**表 1**  SLO调度优化特性参数说明  <a id="table1"></a>

|配置项|取值类型|取值范围|配置说明|
|--|--|--|--|
|stageSelectPolicy|uint32_t|[0,2]|Prefill和Decode选择策略。<br><ul><li>0：prefill优先</li><li>1：吞吐优先</li><li>2：基于TTFT/TPOT时延预测和LLF算法的PD阶段选择算法</li></ul><br>选填，默认值：0。|
|dynamicBatchSizeEnable|bool|<ul><li>true</li><li>false</li></ul>|是否开启动态BatchSize调整算法。<br>选填，默认值：false。|
|prefillExpectedTime|uint32_t|[0,10000]|Prefill阶段Token生成的SLO期望时延。<br>选填，默认值：1500。|
|decodeExpectedTime|uint32_t|[0,10000]|Decode阶段Token生成的SLO期望时延。<br>选填，默认值：50。|


## 执行推理

本章节简单介绍如何使用SLO调度优化功能。

1. 打开MindIE Motor的config.json文件。

    ```bash
    cd {MindIE安装目录}/latest/mindie-service/
    vi conf/config.json
    ```

2. 配置服务化参数。在Server的config.json文件添加“stageSelectPolicy”、“dynamicBatchSizeEnable” 、“prefillExpectedTime”、“decodeExpectedTime”字段（以下加粗部分）， 参数字段解释请参见[表1](#table1)，服务化参数说明请参见[配置参数说明（服务化）](../user_manual/service_parameter_configuration.md)章节，参数配置示例如下。

    ```json
    "stageSelectPolicy" : 2,
    "dynamicBatchSizeEnable" : true,
    "prefillExpectedTime" : 1000,
    "decodeExpectedTime" : 50
    ```

3. 启动服务。

    ```bash
    ./bin/mindieservice_daemon
    ```

4. 以AISBench工具、gsm8k数据集和并发500为例展示调优方式。AISBench工具配置如下，详情请参见《MindIE Motor开发指南》中的“快速入门 > [性能测试](https://gitcode.com/Ascend/MindIE-Motor/blob/dev/docs/zh/user_guide/quick_start.md)”章节。

    ```
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

