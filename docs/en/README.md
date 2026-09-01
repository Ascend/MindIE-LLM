---
hide:
  - navigation
  - toc
---

# Welcome to MindIE-LLM

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-04T12:37:15.277Z pushedAt=2026-06-05T01:09:38.095Z -->

<div style="text-align: center; margin: 0.5rem 0 0.3rem 0; font-family: 'Avenir Next', 'Avenir', 'Century Gothic', 'Segoe UI', sans-serif;">
  <span style="font-size: 4.5rem; font-weight: 300; letter-spacing: 0.02em;">MindIE-LLM</span>
</div>

Mind Inference Engine Large Language Model (MindIE LLM) is a large language model inference component within MindIE. Built on Ascend hardware, it delivers general-purpose LLM inference capabilities along with multi-concurrent request scheduling.

Choose an entry point based on your usage scenario:

- To run model inference using MindIE LLM, it is recommended to start with [Quick Start](user_guide/quick_start/quick_start.md).

- To install and deploy MindIE LLM, it is recommended to start with [Installation Guide](user_guide/install/installation_introduction.md).

- To perform serving deployment and parameter tuning, it is recommended to start with [User Manual](user_guide/user_manual/introduction.md).

- To learn about supported models and features, it is recommended to start with the [Model Support List](user_guide/model_support_list.md) and [Feature Overview](user_guide/feature/README.md).

- To participate in model migration, adaptation, and feature development, it is recommended to start with the [Development Guide](developer_guide/architecture_design/architecture_overview.md).

## Core Capabilities

MindIE LLM delivers high-performance inference capabilities:

- High-throughput serving inference, supporting Continuous Batching and PagedAttention

- Efficient attention KV cache memory management

- Multiple quantization support: W8A8, W8A16, W4A8, FA3, KV cache INT8, etc.

- Multi-dimensional parallelism strategies: tensor parallelism, data parallelism, expert parallelism, context parallelism, sequence parallelism

- Prefill/Decode co-location and KV cache pooling

- SplitFuse block scheduling, asynchronous scheduling, and parallel decoding for latency reduction

MindIE LLM is flexible and easy to use:

- One-click deployment with Docker images and ready to use out of the box

- Mainstream open-source LLMs support

- Compatible with inference APIs from OpenAI, Triton, TGI, vLLM, and others

- Rich model features including MoE, MLA, MTP, Function Call, and Multi-LoRA

- Comprehensive parameter and environment variable system

## Architecture Overview

The overall architecture of MindIE LLM is divided into four layers:

- **Server**: Provides RESTful APIs that are compatible with mainstream inference frameworks such as Triton, OpenAI, TGI, and vLLM.

- **LLM Manager**: Manages system state and task scheduling. It implements request batching based on scheduling policies and manages the unified memory pool for KV Cache.

- **Text Generator**: Handles model configuration, initialization, loading, autoregressive inference, and postprocessing.

- **Modeling**: Provides performance-tuned modules and built-in models, supporting ATB Models.

For details, see [Architecture Overview](developer_guide/architecture_design/architecture_overview.md).

## Related Links

- [Ascend Community](https://www.hiascend.com/)

- [MindIE Image Repository](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)
