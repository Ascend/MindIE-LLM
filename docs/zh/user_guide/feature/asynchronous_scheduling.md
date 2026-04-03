# 异步调度

异步调度是一种利用模型推理耗时掩盖数据处理耗时的调度算法，它的出现解决了同步推理场景下CPU与NPU串行使用导致的时间浪费问题。
在同步推理场景下，一次推理的过程可以按照在CPU/NPU上执行分为以下三个阶段：

-  请求调度与准备阶段（CPU上执行）
-  模型推理与采样阶段（NPU上执行）
-  结果判断与响应阶段（CPU上执行）

其中，CPU与NPU任务由于使用的计算资源不同可以并行执行。为了提升系统整体的资源利用率和吞吐量，MindIE利用了上述特点、使用多线程实现了异步调度模式。
值得注意的是，在该模式下，已经进入EOS（终止推理）状态的请求会被重复计算一次，这会导致NPU计算资源和内存资源有微小的浪费。
因此，该特性通常适用于max_batch_size较大，且输出长度较长的场景。

## 限制与约束

-  支持PD混部和PD分离场景。
-  该特性不能和Look Ahead、Memory Decoding同时使用。
-  该特性暂不支持n、best\_of、use\_beam\_search等与多序列推理相关的后处理参数。

## 执行推理

1. 设置环境变量，打开异步调度功能。

    ```bash
    export MINDIE_ASYNC_SCHEDULING_ENABLE=1
    ```

    > [!NOTE]说明 
    > PD分离部署场景下，请仅在D节点设置环境变量打开异步调度功能。

2. 打开Server的config.json文件。

    ```bash
    cd {MindIE安装目录}/latest/mindie-service/
    vi conf/config.json
    ```

3. 配置服务化参数。服务化参数说明请参见[配置参数说明（服务化）](../user_manual/service_parameter_configuration.md)章节。
4. 启动服务。

    ```bash
    ./bin/mindieservice_daemon
    ```

5. 使用AISBench工具开始调优，AISBench工具详细说明请参见《MindIE Motor开发指南》中的“配套工具 > 性能/精度测试工具”章节。


