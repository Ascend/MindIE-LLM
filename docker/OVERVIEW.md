# MindIE Docker

> English | [中文](./OVERVIEW.zh.md)

Provides automated build scripts and a multi-stage Dockerfile for building MindIE inference service images from compiled packages. The build pipeline covers Python compilation, CANN toolchain installation, PyTorch/torch_npu (PTA) deployment, and MindIE component installation (LLM / SD / Motor), with 8-way parallel downloads for acceleration.

## Quick Reference

- MindIE is maintained by the [MindIE community](https://www.hiascend.com/cn/developer/software/mindie)

- Where to get help

    - [MindIE Image Registry](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)
    - [MindIE-LLM Repository](https://gitcode.com/Ascend/MindIE-LLM)
    - [MindIE-SD Repository](https://gitcode.com/Ascend/MindIE-SD)
    - [MindIE-Motor Repository](https://gitcode.com/Ascend/MindIE-Motor)
    - [Developer Community](https://www.hiascend.com/developer)
    - [Issue Tracker](https://gitcode.com/Ascend/MindIE-LLM/issues)

---

## MindIE

MindIE is Huawei Atlas's inference acceleration suite. It leverages a highly optimized model library and inference engine to boost performance and usability on Atlas hardware. MindIE supports industry-standard model inference, multi-request scheduling, and features like Continuous Batching, PagedAttention, and FlashDecoding to enable high-performance inference. The suite includes:

- **MindIE LLM** — large language model inference
- **MindIE SD** — stable diffusion image generation
- **MindIE Motor** — inference service orchestration

---

## File Overview

| File | Description |
|------|-------------|
| [build.sh](./build.sh) | Main entry point — argument parsing, validation, download orchestration, and build invocation |
| [Dockerfile](./Dockerfile) | Multi-stage Docker build file (7 stages) |
| [modules/config.sh](./modules/config.sh) | Central configuration: URL templates, logging, validation, arch detection, Chip/OS metadata |
| [modules/download.sh](./modules/download.sh) | Download layer: 8 parallel downloads for PTA / Python / CANN / MindIE-LLM / MindIE-SD / MindIE-Motor packages |
| [modules/build_image.sh](./modules/build_image.sh) | Build orchestration layer: image tag computation, Docker build, image export |

---

## Supported Tags and Dockerfile Links

### Tag Specification

Tags follow this format:

```text
mindie:<MindIE-version>-<product-series>-<python-version>-<os>-<arch>
```

| Field | Example | Description |
|-------|---------|-------------|
| `MindIE-version` | `3.0.0` | MindIE version number (drives LLM / SD / Motor) |
| `product-series` | `800I-A2`, `800I-A3`, `300I-Duo` | Target Atlas product series |
| `python-version` | `py3.11` | Python version |
| `os` | `ubuntu24.04`, `openeuler` | Base operating system |
| `arch` | `x86_64`, `aarch64` | CPU architecture |

### Image Registry

MindIE images support base image pre-pulling through a mirror registry:

```text
swr.cn-north-4.myhuaweicloud.com/inference
```

**Full image example:**

```text
mindie:3.0.0-800I-A2-py3.11-ubuntu24.04-x86_64
```

### Product Series Mapping

| Chip Parameter | Product Series | Description |
|----------------|---------------|-------------|
| `310` | `300I-Duo` | Atlas 300I Pro / 300V Pro |
| `910` | `800I-A2` | Atlas 800T A2 / 900 A2 PoD |
| `A3` | `800I-A3` | Atlas 800T A3 |

---

## Build Parameters

Build parameters are passed as command-line arguments to `build.sh`:

| Parameter | Description | Required | Default | Example |
|-----------|-------------|----------|---------|---------|
| `--os` | Server operating system | Yes | — | ubuntu / openeuler |
| `--chip` | Atlas device model | Yes | — | 310 / 910 / A3 |
| `--arch` | System architecture | Yes | — | x86_64 / aarch64 |
| `--mindie` | MindIE version (drives LLM / SD / Motor) | Yes | — | 3.0.0 |
| `--cann` | CANN version | Yes | — | 9.0.0 |
| `--pta-tag` | PTA release tag | Yes | — | v26.0.0-pytorch2.9.0 |
| `--type` | Package type | No | `whl` | whl / run |
| `--python` | Python version | No | `3.11.10` | 3.11.6 |
| `--dry-run` | Validate and show config only | No | `false` | — |

**Note:**

1. CANN version: see [Atlas Community](https://www.hiascend.com/developer/download/community/result)

2. PTA tag: see [Pytorch-NPU Community](https://gitcode.com/Ascend/pytorch/releases)

3. MindIE version: see [MindIE-LLM Community](https://gitcode.com/Ascend/MindIE-LLM/releases) / [MindIE-SD Community](https://gitcode.com/Ascend/MindIE-SD/releases) / [MindIE-Motor Community](https://gitcode.com/Ascend/MindIE-Motor/releases)

---

## Quick Start

### Prerequisites

- Docker must be installed on the host (version ≥ 24.x.x)
- Sufficient disk space for the build directory (~50GB+ including downloads and build cache)
- Access to Atlas OBS mirrors and Huawei Cloud PyPI mirror

---

### Building the MindIE Image

Run the build script from the `docker` directory:

```bash
# Full parameter example (whl package, default Python 3.11.10)
bash build.sh \
    --os=ubuntu \
    --chip=910 \
    --arch=x86_64 \
    --mindie=3.0.0 \
    --cann=9.0.0 \
    --pta-tag=v26.0.0-pytorch2.9.0

# Run package + custom Python version
bash build.sh \
    --os=openeuler \
    --chip=310 \
    --arch=aarch64 \
    --mindie=3.0.0 \
    --cann=9.0.0 \
    --pta-tag=v26.0.0-pytorch2.9.0 \
    --type=run \
    --python=3.11.6

# Dry run: validate parameters only, skip the build
bash build.sh \
    --os=ubuntu \
    --chip=910 \
    --arch=x86_64 \
    --mindie=3.0.0 \
    --cann=9.0.0 \
    --pta-tag=v26.0.0-pytorch2.9.0 \
    --dry-run
```

### Build Pipeline

The build process runs through the following steps in order:

1. **Argument Parsing & Validation** — `build.sh` parses CLI arguments and calls `config.sh` to validate OS/Chip/Arch/Type values.
2. **Parallel Downloads (8-way)** — `download.sh` downloads the following components in parallel:
   - PTA (torch_npu wheel)
   - Python source tarball (Ubuntu only; openEuler skips)
   - CANN Toolkit
   - CANN NNAL
   - CANN Kernels (chip-specific operator package)
   - MindIE-LLM package (whl or run)
   - MindIE-SD package (whl only)
   - MindIE-Motor package (whl only)
3. **Docker Multi-Stage Build** — `build_image.sh` invokes the `Dockerfile` through 7 stages:
   - **Stage 1a (base-ubuntu):** Ubuntu 24.04 + compile Python from source
   - **Stage 1b (base-openeuler):** OpenEuler 24.03 + pre-installed Python
   - **Stage 2 (base):** Dynamic OS selection, import all downloaded packages
   - **Stage 3 (cann):** Install CANN Toolkit + Kernels + NNAL
   - **Stage 4 (pta):** Install PyTorch + torch_npu
   - **Stage 4.5 (mindstudio):** Install dev tools (git, cmake, gcc, ffmpeg, etc.)
   - **Stage 5 (mindie):** Install MindIE components (LLM / SD / Motor)
4. **Image Export** — Save the built image as a `.tar.gz` file in the `output/` directory.

### Dockerfile Multi-Stage Build Diagram

```text
base-ubuntu ──┐
              ├──> base ──> cann ──> pta ──> mindstudio ──> mindie
base-openeuler┘
```

---

## Download Sources

| Component | Source |
|-----------|--------|
| MindIE-LLM | `https://gitcode.com/Ascend/MindIE-LLM/releases/download` |
| MindIE-SD | `https://gitcode.com/Ascend/MindIE-SD/releases/download` |
| MindIE-Motor | `https://gitcode.com/Ascend/MindIE-Motor/releases/download` |
| PTA (torch_npu) | `https://gitcode.com/Ascend/pytorch/releases/download` |
| CANN | `https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN` |
| Python Source | `https://mirrors.huaweicloud.com/python` |

---

## Supported Hardware

| Chip Series | Product Examples | Architecture |
|-------------|-----------------|--------------|
| Atlas 910 | Atlas 800T A2, Atlas 900 A2 PoD | ARM64 / x86_64 |
| Atlas A3 | Atlas 800T A3 | ARM64 / x86_64 |
| Atlas 310 | Atlas 300I Pro, Atlas 300V Pro | ARM64 / x86_64 |

---

## Container Environment Variables

The following key environment variables are set during the Docker build:

| Variable | Description |
|----------|-------------|
| `ASCEND_TOOLKIT_HOME` | CANN toolchain installation path |
| `MINDIE_LLM_HOME_PATH` | MindIE-LLM service installation path |
| `MIES_INSTALL_PATH` | MindIE-Motor (mindie-service) installation path |
| `ATB_SPEED_HOME_PATH` | ATB-LLM acceleration library path |
| `MINDIE_LLM_CONTINUOUS_BATCHING` | Continuous batching toggle (default 1) |
| `ASCEND_GLOBAL_LOG_LEVEL` | Global log level (default 3) |

---

## License

View the MindIE [license information](../LICENSE.md).

As with all container images, pre-installed software packages (Python, system libraries, etc.) may be subject to their own licenses.
