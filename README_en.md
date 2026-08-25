<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-04T12:38:26.960Z pushedAt=2026-06-05T02:45:37.309Z -->

<h1 align="center" style="font-size: 3.5rem;">MindIE-LLM</h1>

<p align="center"><b>Ascend Large Language Model Inference Engine</b></p>

<div align="center">

[🏠 Ascend Community](https://www.hiascend.com/) |
[📖 Documentation Center](https://mindie-llm-doc.readthedocs.io/zh-cn/latest/) |
[📅 Community Meeting](https://meeting.ascend.osinfra.cn/?sig=sig-MindIE-LLM) |
[💬 Issues](https://gitcode.com/Ascend/MindIE-LLM/issues)

</div>

<div align="center">

[![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/verylucky01/MindIE-LLM)&nbsp;&nbsp;&nbsp;&nbsp;
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/verylucky01/MindIE-LLM)

</div>

English | [简体中文](./README.md)

## 📢 Latest News

- [2026/04] 📖 The documentation site is now available! Welcome to the [MindIE-LLM Documentation Center](https://mindie-llm-doc.readthedocs.io/zh-cn/latest/) to read the complete documentation online.

- [2025/12] MindIE LLM is officially open-sourced and available to the public! [Meeting Calendar](https://meeting.ascend.osinfra.cn/?sig=sig-MindIE-LLM)

## 🚀 Overview

**MindIE LLM** is an inference acceleration suite for large language models on Ascend. It is designed to specifically enhance the inference performance and usability of models on Ascend hardware through a deeply optimized model library and inference optimizer. MindIE LLM delivers general-purpose LLM inference on Ascend hardware, featuring multi-request scheduling and acceleration techniques such as continuous batching, PagedAttention, and FlashDecoding to meet high-performance inference demands.

## 🔍 Directory Structure

```plaintext
├── mindie_llm                                     # Main Python inference framework
│   ├── connector                                  # Request access layer
│   ├── text_generator                             # Core inference engine
│   ├── modeling                                   # Model abstraction layer
│   ├── runtime                                    # Runtime compilation & model loading
│   ├── utils                                      # Utilities: logging, tensors, profiling, validation
├── examples                                       # Sample code
├── docs                                           # Project documentation
├── src                                            # Core C++ engine
│   ├── engine                                     # Main LLM engine logic (scheduling/execution)
│   ├── scheduler                                  # Schedulers: FCFS, PDDS, Layerwise
│   ├── block_manager                              # KV cache block management: LRU, Prefix Cache, CoW
│   ├── llm_manager                                # Python/C++ bridge API
│   ├── server                                     # gRPC/HTTP server endpoints
│   ├── utils                                      # Basic utilities: shared memory, encryption, logging, ID generation
│   ├── include                                    # Public headers
├── scripts                                        # Build & deployment scripts
├── tools                                          # Auxiliary tools
├── tests                                          # Tests
├── CMakeLists.txt                                 # CMake build configuration
├── README.md
```

## 📢 Version Description

| MindIE Version&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| CANN Version&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|
|:----------------------------|:----------------------------|
| 2.3.0 | 8.5.0 |

## ⚡️ Environment Deployment

- To install MindIE LLM via a software package or image, see [Installation Guide](./docs/en/user_guide/install/menu_install.md).

- To build and install MindIE LLM by pulling the latest code, see [Build and Installation Guide](./docs/en/developer_guide/build_guide_llm.md).

## ⚡️ Quick Start

To quickly experience the full workflow of model inference using MindIE, see [Quick Start](./docs/en/user_guide/model_support_list.md).

## 📝 Documentation

- Model Support List

  - [Repository Model Support List](./docs/en/user_guide/model_support_list.md): Prioritize using this list, which provides the complete set of models that have been fully tested and verified for support, as well as those with functional support only, for the current version.

  - [Ascend Community Model Support List](https://www.hiascend.com/software/mindie/modellist): Models that have been fully tested and verified for support in the current version.

- [Feature Introduction](./docs/en/user_guide/feature/README.md): Inference features supported by MindIE LLM.

- [LLM User Guide](./docs/en/user_guide/user_manual/menu_user_manual.md): MindIE LLM user guide, including inference parameter configuration, online and offline inference, parameter tuning, and more.

## 📝Contribution Statement

1. Submitting bug reports: If you discover a non-security vulnerability in MindIE LLM, first search the existing Issues in the MindIE LLM repository to avoid duplicates. If no related issue is found, you may open a new one. For security vulnerabilities, do not disclose them publicly—refer to the security issue handling process instead. Always include complete information when submitting a bug report.

2. Security issue handling: For the handling of security issues in this project, notify the core project personnel via email for confirmation.

3. Resolve existing issues: By reviewing the repository's Issues list, you can find information about problems that need to be addressed and try to resolve them.

4. Requesting a new feature: Use the `Feature` label in Issues. We regularly review and prioritize submissions for development.

5. Start contributing.

 <br>a. Fork the repository of this project.
 <br>b. Clone it locally.
 <br>c. Create a development branch.
 <br>d. Run local tests. Before committing, ensure all unit tests pass, including new ones added for your changes.
 <br>e. Commit the code.
 <br>f. Create a new Pull Request.
 <br>g. Address review feedback. Revise your code as requested and push updates. Multiple rounds may be needed.
 <br>h. Once your PR has enough approvals, a Committer will perform the final review.
 <br>i. After your PR is approved and all tests pass, the CI system will merge it into the main branch of the project.

For more contribution-related documents, See [Contribution Guide](./contributing_en.md).

## 📝 Disclaimer

Copyright © 2025-2026 MindIE Project.

Your reproduction, use, modification, and distribution of "this document" are governed by the Creative Commons Attribution-ShareAlike 4.0 International Public License (hereinafter referred to as "CC BY-SA 4.0"). For ease of understanding, you may visit [https://creativecommons.org/licenses/by-sa/4.0/](https://creativecommons.org/licenses/by-sa/4.0/) for a summary (but not a substitute) of CC BY-SA 4.0. For the full text of CC BY-SA 4.0, please visit the following URL: [https://creativecommons.org/licenses/by-sa/4.0/legalcode](https://creativecommons.org/licenses/by-sa/4.0/legalcode).

## 🌟 Related Information

- [Security Statement](./security.md)

- [LICENSE](./LICENSE_en.md)
