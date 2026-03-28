# 简介

## 概述

**MindIE LLM**（Mind Inference Engine for Large Language Models）是昇腾的大语言模型（Large Language Model，LLM）推理加速套件，旨在通过深度优化的模型库和推理优化器，专门提升大模型在昇腾硬件上的推理性能和易用性。MindIE LLM基于昇腾硬件，提供业界通用大模型推理能力，多并发请求的调度，包含Continuous Batching、PagedAttention、FlashDecoding等加速特性，使能用户高性能推理需求。

MindIE LLM 主要对外提供 **C++ 与 Python API**（Application Programming Interface），包括大模型推理、并发请求调度和 LLM Manager API 等，便于用户在业务系统中集成与调用。

## MindIE LLM架构图

**图1** MindIE LLM架构图

![](./figures/mindie_llm_architecture_diagram.png)

    - Engine：负责将scheduler，executor，worker等协同串联起来，利用组件间的协同，实现多场景下请求的推理处理能力。
    - Scheduler: 在1个DP域内，将多条请求在Prefill或者Decode阶段组成batch，实现计算和通信的充分利用。
    - Block manager：管理在DP内的kv资源，支持池化后，支持对offload的kv位置感知。
    - Executor：将调度完成的信息分发给Text Generator模块。支持跨机、跨卡的任务下发。

- **Server**：推理服务层，对外提供模型推理的服务化能力与统一接入能力。Endpoint 面向推理服务开发者提供 RESTful 接口，同时，Endpoint 负责推理服务化协议与接口的封装，并兼容 Triton/OpenAI/TGI/vLLM 等主流推理框架的请求接口。

- **LLM Manager**：负责请求状态管理与任务调度。其基于调度策略将用户请求组成 Batch，并通过统一性内存池管理键值缓存（KV Cache）。LLM Manager 汇总并返回推理结果，同时提供状态记录与查询接口。

    - LLM Manager Interface：MindIE-LLM 推理引擎对外暴露的接口层，用于对接上层服务调用与能力集成。
    - Engine：负责对 Scheduler、Executor、Worker 等组件进行编排与串联。通过组件间的协同，Engine 为不同推理场景提供统一的请求处理与执行能力。
    - Scheduler：在一个 DP（Data Parallel，数据并行）域内，将多条请求在 Prefilling（预填充）或 Decoding（解码）阶段组成 Batch。该策略用于提升计算与通信资源的利用率，从而提高整体吞吐与效率。
    - Block Manager：管理 DP 域内的 KV Cache 资源，并支持池化（Pooling）管理以提升内存复用效率。同时，Block Manager 支持对 Offload（卸载到 Host 端或外部存储）的 KV Cache 进行位置感知与索引管理。
    - Executor：将调度阶段生成的执行计划与元信息下发至 Text Generator 模块。Executor 支持分布式推理场景下的任务派发，包括跨机与跨卡执行。

- **Text Generator**：负责模型配置、初始化与加载，并实现自回归推理流程及结果后处理。其向 LLM Manager 提供统一的自回归推理接口，并支持并行解码能力的插件化扩展与运行。

    - Preprocess：将调度后的任务转换为模型可直接消费的输入表示。
    - Generator：对模型运行过程进行抽象封装，覆盖前向计算、状态更新以及自回归式解码等核心执行逻辑。
    - Sampler：基于模型输出的 Logits 完成 Token 选择（如贪心搜索、束搜索、Top-p 采样、基于温度的采样等策略）、停止条件判断，并负责上下文状态的更新与必要的清理（如缓存回收）。

- **Modeling**：提供经过性能调优的算子模块与内置模型实现，支持 ATB Models（Ascend Transformer Boost Models）。

    - 内置模块包括 Attention、Embedding、ColumnLinear、RowLinear、MLP（Multi-Layer Perceptron，多层感知机）与 MoE（Mixture of Experts）。这些模块支持对权重（Weight）进行在线 Tensor 切分与加载。

    - 内置模型基于上述模块完成完整网络构建与组合，并支持 Tensor 切分。同时，内置模型支持多种量化方式。用户也可参考示例，使用内置模块自行构建并定制模型结构。

    - 模型完成组网后将进入编译与优化流程，最终生成可在昇腾 NPU 设备上进行加速推理的可执行计算图。
