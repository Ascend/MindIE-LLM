# MindIE Docker

> [English](./OVERVIEW.md) | 中文

提供自动化构建脚本和多阶段 Dockerfile，支持从编译包构建 MindIE 推理服务镜像。构建流程涵盖 Python 编译、CANN 工具链安装、PyTorch/torch_npu（PTA）部署、以及 MindIE 组件安装（LLM / SD / Motor），并支持 8 路并行下载加速。

## 快速参考

- MindIE 由 [MindIE community](https://www.hiascend.com/cn/developer/software/mindie) 维护

- 从哪里获取帮助

    - [MindIE 镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)
    - [MindIE-LLM 仓库](https://gitcode.com/Ascend/MindIE-LLM)
    - [MindIE-SD 仓库](https://gitcode.com/Ascend/MindIE-SD)
    - [MindIE-Motor 仓库](https://gitcode.com/Ascend/MindIE-Motor)
    - [昇腾开发者社区](https://www.hiascend.com/developer)
    - [问题反馈](https://gitcode.com/Ascend/MindIE-LLM/issues)

---

## MindIE

**MindIE** 是昇腾的推理加速套件，旨在通过深度优化的模型库和推理优化器，专门提升大模型在昇腾硬件上的推理性能和易用性。MindIE 基于昇腾硬件，提供业界通用大模型推理能力，多并发请求的调度，包含 Continuous Batching、PagedAttention、FlashDecoding 等加速特性，使能用户高性能推理需求。套件包含：

- **MindIE LLM** — 大语言模型推理
- **MindIE SD** — Stable Diffusion 图像生成
- **MindIE Motor** — 推理服务编排

---

## 文件说明

| 文件 | 说明 |
|------|------|
| [build.sh](./build.sh) | 主入口脚本，负责参数解析、校验、下载编排与构建调用 |
| [Dockerfile](./Dockerfile) | 多阶段 Docker 构建文件（7 个阶段） |
| [modules/config.sh](./modules/config.sh) | 集中配置：URL 模板、日志、校验、架构检测、Chip/OS 元数据 |
| [modules/download.sh](./modules/download.sh) | 下载层：8 路并行下载 PTA / Python / CANN / MindIE-LLM / MindIE-SD / MindIE-Motor 包 |
| [modules/build_image.sh](./modules/build_image.sh) | 构建编排层：镜像 Tag 计算、Docker 构建、镜像导出 |

---

## 支持的 Tag 及 Dockerfile 链接

### Tag 规范

Tag 遵循以下格式：

```text
mindie:<MindIE版本>-<产品系列>-<python版本>-<操作系统>-<架构类型>
```

| 字段 | 示例值 | 说明 |
|------|--------|------|
| `MindIE版本` | `3.0.0` | MindIE 版本号（驱动 LLM / SD / Motor） |
| `产品系列` | `800I-A2`、`800I-A3`、`300I-Duo` | 目标昇腾产品系列 |
| `python版本` | `py3.11` | Python 版本 |
| `操作系统` | `ubuntu24.04`、`openeuler` | 基础操作系统 |
| `架构类型` | `x86_64`、`aarch64` | CPU 架构类型 |

### 镜像仓库地址

MindIE 镜像支持通过镜像仓库加速拉取基础镜像：

```text
swr.cn-north-4.myhuaweicloud.com/inference
```

**完整镜像示例：**

```text
mindie:3.0.0-800I-A2-py3.11-ubuntu24.04-x86_64
```

### 产品系列映射

| Chip 参数 | 产品系列 | 说明 |
|-----------|----------|------|
| `310` | `300I-Duo` | Atlas 300I Duo |
| `910` | `800I-A2` | Atlas 800I A2 |
| `A3` | `800I-A3` | Atlas 800I A3 |

---

## 构建参数

构建脚本通过命令行参数传递，以下为参数说明：

| 参数 | 说明 | 是否必填 | 默认值 | 示例值 |
|------|------|----------|--------|--------|
| `--os` | 服务器操作系统 | 是 | — | ubuntu / openeuler |
| `--chip` | 昇腾设备型号 | 是 | — | 310 / 910 / A3 |
| `--arch` | 系统架构 | 是 | — | x86_64 / aarch64 |
| `--mindie` | MindIE 版本号（驱动 LLM / SD / Motor） | 是 | — | 3.0.0 |
| `--cann` | CANN 版本号 | 是 | — | 9.0.0 |
| `--pta-tag` | PTA 发布 Tag | 是 | — | v26.0.0-pytorch2.9.0 |
| `--type` | 包类型 | 否 | `whl` | whl / run |
| `--python` | Python 版本 | 否 | `3.11.10` | 3.11.6 |
| `--dry-run` | 仅校验参数并展示配置 | 否 | `false` | — |

**注意：**

1. cann 版本号参见[昇腾社区](https://www.hiascend.com/developer/download/community/result)

2. pta-tag 参见 [Pytorch-NPU 社区](https://gitcode.com/Ascend/pytorch/releases)

3. mindie 版本号参见 [MindIE-LLM 社区](https://gitcode.com/Ascend/MindIE-LLM/releases) / [MindIE-SD 社区](https://gitcode.com/Ascend/MindIE-SD/releases) / [MindIE-Motor 社区](https://gitcode.com/Ascend/MindIE-Motor/releases)

---

## 快速开始

### 前置要求

- 宿主机上需要安装 Docker（版本 ≥ 24.x.x）
- 构建目录需要有足够的磁盘空间（下载包 + 构建缓存约 50GB+）
- 可以访问昇腾 OBS 镜像源和华为云 PyPI 镜像

---

### 构建 MindIE 镜像

在 `docker` 目录下执行构建脚本：

```bash
# 完整参数示例（whl 包，默认 Python 3.11.10）
bash build.sh \
    --os=ubuntu \
    --chip=910 \
    --arch=x86_64 \
    --mindie=3.0.0 \
    --cann=9.0.0 \
    --pta-tag=v26.0.0-pytorch2.9.0

# 使用 run 包 + 自定义 Python 版本
bash build.sh \
    --os=openeuler \
    --chip=310 \
    --arch=aarch64 \
    --mindie=3.0.0 \
    --cann=9.0.0 \
    --pta-tag=v26.0.0-pytorch2.9.0 \
    --type=run \
    --python=3.11.6

# 仅校验参数，不执行构建
bash build.sh \
    --os=ubuntu \
    --chip=910 \
    --arch=x86_64 \
    --mindie=3.0.0 \
    --cann=9.0.0 \
    --pta-tag=v26.0.0-pytorch2.9.0 \
    --dry-run
```

### 构建流程

构建过程依次完成以下步骤：

1. **参数解析与校验** — `build.sh` 解析命令行参数，调用 `config.sh` 校验 OS/Chip/Arch/Type 合法性。
2. **并行下载（8 路）** — `download.sh` 并行下载以下组件：
   - PTA（torch_npu wheel）
   - Python 源码包（仅 Ubuntu，openEuler 跳过）
   - CANN Toolkit
   - CANN NNAL
   - CANN Kernels（芯片相关算子包）
   - MindIE-LLM 包（whl 或 run）
   - MindIE-SD 包（仅 whl）
   - MindIE-Motor 包（仅 whl）
3. **Docker 多阶段构建** — `build_image.sh` 调用 `Dockerfile` 执行 7 阶段构建：
   - **Stage 1a (base-ubuntu):** Ubuntu 24.04 + 从源码编译 Python
   - **Stage 1b (base-openeuler):** OpenEuler 24.03 + 预装 Python
   - **Stage 2 (base):** 动态 OS 选择，导入所有下载包
   - **Stage 3 (cann):** 安装 CANN Toolkit + Kernels + NNAL
   - **Stage 4 (pta):** 安装 PyTorch + torch_npu
   - **Stage 4.5 (mindstudio):** 安装开发工具（git, cmake, gcc, ffmpeg 等）
   - **Stage 5 (mindie):** 安装 MindIE 组件（LLM / SD / Motor）
4. **镜像导出** — 将构建产物保存为 `output/` 目录下的 `.tar.gz` 文件。

### Dockerfile 多阶段构建图

```text
base-ubuntu ──┐
              ├──> base ──> cann ──> pta ──> mindstudio ──> mindie
base-openeuler┘
```

---

## 下载源

| 组件 | 下载源 |
|------|--------|
| MindIE-LLM | `https://gitcode.com/Ascend/MindIE-LLM/releases/download` |
| MindIE-SD | `https://gitcode.com/Ascend/MindIE-SD/releases/download` |
| MindIE-Motor | `https://gitcode.com/Ascend/MindIE-Motor/releases/download` |
| PTA (torch_npu) | `https://gitcode.com/Ascend/pytorch/releases/download` |
| CANN | `https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN` |
| Python 源码 | `https://mirrors.huaweicloud.com/python` |

---

## 支持的硬件

| 芯片系列 | 产品示例 | 架构 |
|----------|----------|------|
| 昇腾 910 | Atlas 800I A2 | ARM64 |
| 昇腾 A3 | Atlas 800I A3 | ARM64 |
| 昇腾 310 | Atlas 300I Duo | ARM64 / x86_64 |

---

## 环境变量（容器内）

构建过程中 Dockerfile 会设置以下关键环境变量：

| 变量 | 说明 |
|------|------|
| `ASCEND_TOOLKIT_HOME` | CANN 工具链安装路径 |
| `MINDIE_LLM_HOME_PATH` | MindIE-LLM 服务安装路径 |
| `MIES_INSTALL_PATH` | MindIE-Motor（mindie-service）安装路径 |
| `ATB_SPEED_HOME_PATH` | ATB-LLM 加速库路径 |
| `MINDIE_LLM_CONTINUOUS_BATCHING` | 连续批处理开关（默认 1） |
| `ASCEND_GLOBAL_LOG_LEVEL` | 全局日志级别（默认 3） |

---

## 许可证

查看 MindIE 的[许可证信息](../LICENSE.md)。

与所有容器镜像一样，预装软件包（Python、系统库等）可能受其自身许可证约束。
