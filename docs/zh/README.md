---
hide:
  - navigation
  - toc
---

# 欢迎使用 MindIE-LLM

<div style="text-align: center; margin: 0.5rem 0 0.3rem 0; font-family: 'Avenir Next', 'Avenir', 'Century Gothic', 'Segoe UI', sans-serif;">
  <span style="font-size: 4.5rem; font-weight: 300; letter-spacing: 0.02em;">MindIE-LLM</span>
</div>

MindIE LLM（Mind Inference Engine Large Language Model）是 MindIE 下的大语言模型推理组件，基于昇腾硬件提供业界通用大模型推理能力，同时提供多并发请求的调度功能。

根据你的使用场景选择入口：

- 使用 MindIE LLM 运行模型推理，推荐从 [快速入门](user_guide/quick_start/quick_start.md) 开始
- 安装部署 MindIE LLM，推荐从 [安装指南](user_guide/install/installation_introduction.md) 开始
- 进行服务化部署和参数调优，推荐从 [使用手册](user_guide/user_manual/introduction.md) 开始
- 了解支持的模型和特性，推荐从 [模型支持列表](user_guide/model_support_list.md) 和 [特性总览](user_guide/feature/README.md) 开始
- 参与模型迁移适配与特性开发，推荐从 [开发指南](developer_guide/architecture_design/architecture_overview.md) 开始

## 核心能力

MindIE LLM 具备高性能推理能力：

- 高吞吐服务化推理，支持 Continuous Batching 和 PagedAttention
- 高效的注意力 KV Cache 显存管理
- 多种量化支持：W8A8、W8A16、W4A8 混合精度、FA3 量化、KV Cache INT8 等
- 多维并行策略：张量并行、数据并行、专家并行、上下文并行、序列并行
- Prefill/Decode 混合部署与 KV Cache 池化
- SplitFuse 分块调度、异步调度、并行解码降低时延

MindIE LLM 灵活易用：

- Docker 镜像一键部署，开箱即用
- 支持主流开源大语言模型
- 兼容 OpenAI / Triton / TGI / vLLM 等推理框架请求接口
- MoE、MLA、MTP、Function Call、Multi-LoRA 等丰富模型特性
- 完善的参数配置和环境变量体系

## 架构概览

MindIE LLM 总体架构分为四层：

- **Server**：推理服务端，提供 RESTful 接口，支持 Triton/OpenAI/TGI/vLLM 主流推理框架请求接口
- **LLM Manager**：负责状态管理及任务调度，基于调度策略实现请求组 batch，统一内存池管理 KV Cache
- **Text Generator**：负责模型配置、初始化、加载、自回归推理流程、后处理
- **Modeling**：提供性能调优后的模块和内置模型，支持 ATB Models

详见 [架构概览](developer_guide/architecture_design/architecture_overview.md)。

## 相关链接

- [昇腾社区](https://www.hiascend.com/)
- [MindIE 镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)
