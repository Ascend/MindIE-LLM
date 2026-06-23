# Guide to KV Cache Pooling

## Overview

In current LLM inference systems, KV cache is a widely adopted mechanism. Building on it, prefix cache significantly reduces computation time during the prefill phase when a cache hit occurs. However, the prefix cache uses only on-chip memory by default, which has limited capacity and cannot cache a large amount of prefix information. To address this issue, the KV cache pooling feature is used to extend the storage hierarchy, which allows larger-capacity storage media to be added to the prefix cache pool, thereby breaking the capacity limit of the on-chip memory. The KV cache pooling feature can effectively improve the prefix cache hit ratio and significantly reduces the cost of LLM inference.

## Usage Guide

The KV cache pooling feature depends on the prefix cache feature. Additionally, configure the KV cache pooling feature via the following fields in `BackendConfig` of MindIE's `config.json`:

```json
"kvPoolConfig" : {"backend":"", "configPath":""}
```

Or

```json
"kvPoolConfig" : {"backend":"", "configPath":"", "asyncWrite": true}
```

Configuration description:

- `backend`: specifies the pooling backend to be used.
- `configPath`: specifies the path to the configuration file required by the pooling backend.
- `asyncWrite`: specifies whether to enable asynchronous pooling write. The value can be `true` or `false`. If this field is not set, the default value `false` is used, indicating synchronous write.

If the prefix cache feature is enabled and the preceding fields are configured, the KV cache pooling feature is enabled. Different pooling backends need to be installed separately.

## Supported Pooling Backends

### Mooncake

<details>

#### 1. Installation from Source Code

**Step 1:** Install Mooncake from source by following the official Build Guide. Mooncake supports Ascend Direct Transport, an NPU-aware transfer mechanism (see documentation). To enable it, enable `USE_ASCEND_DIRECT` during compilation.

```shell
git clone https://github.com/kvcache-ai/Mooncake.git
cd Mooncake
mkdir build
cd build
cmake -DUSE_ASCEND_DIRECT=ON -DBUILD_SHARED_LIBS=ON -DBUILD_UNIT_TESTS=OFF ..
make -j
make install
```

**Step 2:** After compiling and installing Mooncake Ascend Direct Transport, copy the compiled artifacts to the installation path specified during `make install`. Example using `/usr/local/lib/python3.11/site-packages/mooncake`:

```bash
cp mooncake-common/src/libmooncake_common.so /usr/local/lib/python3.11/site-packages/mooncake
cp mooncake-transfer-engine/src/libtransfer_engine.so /usr/local/lib/python3.11/site-packages/mooncake
cp mooncake-store/src/libmooncake_store.so /usr/local/lib/python3.11/site-packages/mooncake
```

**Step 3:** Copy `/etc/hccn.conf` from the host to `/etc/hccn.conf` inside the container—Mooncake Ascend Direct Transport requires this configuration file.

**Step 4:** Check whether the installation is successful. If no error information is displayed, the installation is successful.

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.11/site-packages/mooncake
mooncake_master --port 12345
```

##### 2. Preparing the Mooncake Client Configuration File

To use Mooncake Ascend Direct Transport, you need to create a Mooncake client configuration file. Refer to the official Mooncake Store configuration guide and the Mooncake Ascend Direct Transport documentation. For example, create a `mooncake.json` file as shown below.

```json
{
    "local_hostname": "localhost",
    "metadata_server": "P2PHANDSHAKE",
    "global_segment_size": 268435456,
    "protocol": "ascend",
    "device_name": "",
    "master_server_address": "master_server_ip:50051",
    "use_ascend_direct": true
}
```

Special configurations (for details about the parameters, see the official Mooncake description):

- `metadata_server`: Set it to `P2PHANDSHAKE`.
- `protocol`: Set it to `ascend`.
- `use_ascend_direct`: Set it to `true`.

##### 3. Usage in MindIE

After completing the setup described in the [Mooncake client configuration](#2-preparing-the-mooncake-client-configuration-file), set the `configPath` field to the Mooncake client configuration file path and specify `backend` as `mooncake`. Then, start the Mooncake master server on terminal 1.

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.11/site-packages/mooncake
mooncake_master --port 12345 --eviction_high_watermark_ratio 0.8 --eviction_ratio 0.05 --rpc_thread_num 128
```

`eviction_high_watermark_ratio` and `eviction_ratio` are parameters of the eviction policy. For details, see the official Mooncake eviction policy description. `rpc_thread_num` indicates the number of concurrent client connections processed by the master. You are advised to increase the value of this parameter to efficiently process concurrent requests.

Start the MindIE service on terminal 2 and configure the following environment variables:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.11/site-packages/mooncake
export ASCEND_BUFFER_POOL=4:8
```

In pooling scenarios, you are advised to enable jemalloc optimization.

```bash
export LD_PRELOAD="path_to_file/libjemalloc.so:$LD_PRELOAD"
```

For details about the configuration, see the official Mooncake Ascend Direct Transport Important Notes.

</details>

## Constraints

The asynchronous write feature is currently supported only for the Qwen dense model (non-MOE) and DeepSeek models (V3/V3.1/R1). When this feature is enabled, it can be combined with the following features. Enabling it alongside other features or models may cause unexpected errors.

- Qwen Dense: asynchronous scheduling, prefix cache, function call, thinking analysis, and YaRN
- DeepSeek V3/V3.1/R1: asynchronous inference, prefix cache, context parallelism, and sequence parallelism

The following features are not supported:

- SplitFuse, Micro Batch, and Multi-Lora

## Declaration

- The pooling backends referenced in this repository are provided solely as non-commercial examples. If you choose to use them, you are responsible for complying with their respective licenses. Huawei assumes no liability for any infringement disputes arising from such use.
- When using the Mooncake pooling backend, please be aware that it currently transmits data in plaintext between the Mooncake Master Server and its clients. For production deployments, ensure that the relevant IPs and ports are not exposed to the public Internet and restrict access to trusted network ranges to mitigate potential security risks
