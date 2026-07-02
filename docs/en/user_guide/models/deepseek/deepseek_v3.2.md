# DeepSeek-V3.2 Model Deployment Guide

## Overview

- [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) is a large language model (LLM) based on a Mixture-of-Experts (MoE) architecture, developed and released by DeepSeek. By introducing the DeepSeek Sparse Attention (DSA) mechanism, DeepSeek redefines the efficiency standards of LLMs. It can complete tasks such as text generation, code writing and explanation, and mathematical reasoning through natural language interaction.
- DeepSeek-V3.2 exhibits superior reasoning capabilities. It employs the dynamic sparse attention (DSA) mechanism to reduce the computational complexity of attention from O(L²) to O(Lk), where L is the sequence length and k (k ≪ L) is the number of selected tokens. It significantly reduces computational load and memory footprint, achieving substantial improvements in inference efficiency while preserving baseline performance.
- MindIE-LLM achieves high-performance inference of DeepSeek-V3.2 on NPU, utilizing the aclgraph graph-mode approach to deliver optimal inference performance.

---

## Feature Matrix

**Table 1** Hardware support

|Model|Atlas 800I A2|Atlas 800I A3|Atlas 300I Duo inference card|
|:-----:|:--------:|:-----------:|:------------------:|
|DeepSeek-V3.2|Four-server 32-card deployment|Two-server 16-card deployment|❌|

**Table 2** Floating-point and quantization

|     Model     | W8A8 Quantization| Sparse Quantization| W4A8 Quantization| KV Cache Quantization| FA3 Quantization|
| :-----------: | :-------: | :------: | :-------: | :-----------: | :------: |
| DeepSeek-V3.2 |     ✅     |    ❌     |     ❌     |       ❌       |    ❌     |

**Table 3** Other features

| Model| Multi-Lora | Load Balancing| Data Parallel | Context Parallel | Tensor Parallel | Expert Parallel | Flash Comm v1 | Asynchronous Scheduling| Chunked Prefill | SLO Scheduling Tuning| Micro Batch | MTP | Prefix Cache | KV Cache Pooling| Function Call | Thinking Analysis| Prefill-Decode Disaggregation|
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----:  | :-----: |
| DeepSeek-V3.2 | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌  | ✅ | ✅ | ✅ |

**Table 4** Feature combination

| Feature| Data Parallel | Context Parallel | W8A8 Quantization| Asynchronous Scheduling| Chunked Prefill | MTP  | Prefix Cache | Function Call | Thinking Analysis| Prefill-Decode Disaggregation|
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| Data Parallel    | ✅ ||||||||||
| Context Parallel | ❌ | ✅ |||||||||
| W8A8 Quantization       | ✅ | ✅ | ✅ ||||||||
| Asynchronous Scheduling         | ✅ | ❌ | ✅ | ✅ |||||||
| Chunked Prefill  | ✅ | ❌ | ✅ | ✅ | ✅ ||||||
| MTP             | ✅ | ✅ | ✅ | ✅ | Co-location ❌<br>Prefill-decode disaggregation ✅| ✅ |||||
| Prefix Cache    | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ ||||
| Function Call   | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |||
| Thinking Analysis         | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ ||
| Prefill-Decode Disaggregation         | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## Environment Setup

### Weight Quantization

You can directly download the W8A8 quantized weights of the DeepSeek-V3.2 model from Modelers, or use the msModelSlim quantization tool to perform quantization on floating-point weights.

#### Downloading Ascend quantized weights

Download the native Ascend W8A8 quantized weights from the Modelers open-source community:

- [DeepSeek-V3.2-w8a8-mtp-QuaRot](https://modelers.cn/models/Eco-Tech/DeepSeek-V3.2-w8a8-mtp-QuaRot)

#### Using msModelSlim to Generate quantized weights

Use [msModelSlim](https://gitcode.com/Ascend/msmodelslim) to generate quantized weights. For details, see [Quick Quantization Guide](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/en/feature_guide/quick_quantization_v1/usage.md). The following describes the quantization methods and the corresponding commands.

**Quantization strategy**

The W8A8 quantization of DeepSeek-V3.2 uses a multi-stage pipeline to perform the following steps in sequence:

1. **QuaRot (rotation quantization)**: Applies mathematical rotations to eliminate activation outliers, optimizing the numerical distribution for subsequent quantization.
2. **Flex Smooth Quant (outlier smoothing)**: Performs post-rotation outlier smoothing across `norm-linear` and `ov` (shared expert) subgraphs.
3. **Linear layer quantization**: Different quantization strategies are used for the Attention and MoE modules:

| Module| Quantization Method| Activation Quantization| Weight Quantization| Description|
|------|----------|-----------|---------|------|
| Attention (self_attn)| W8A8 (static quantization)| per_tensor, int8, asymmetric| per_channel, int8, symmetric| Exclude `kv_b_proj`, `wq_b`, `wk`, and `weights_proj`.|
| MLP/MoE experts| W8A8 (dynamic quantization)| per_token, int8, symmetric| per_channel, int8, symmetric| Exclude the gate layer.|

**Quantization command example:**

Generate DeepSeek-V3.2 W8A8-QuaRot quantized weights:

```shell
msmodelslim quant \
  --model_path ${MODEL_PATH} \     # Path to the model weights
  --save_path ${SAVE_PATH} \       # Path to the quantized weights
  --model_type DeepSeek-V3.2 \     # Fixed to DeepSeek-V3.2
  --quant_type w8a8 \
  --trust_remote_code True
```

### Software Environment

1. For details about how to prepare the environment for the image/physical machine/container installation scenario, see [Environment Setup](../../install/environment_preparation.md).
2. In the multi-server scenario, configure the `rank_table_file.json` file by referring to [Rank Table File Configuration Guide](../../user_manual/rank_table_file_configuration.md).

- After the `rank_table_file.json` file is configured, change the permission to `640`.

- If you deploy the service as a common user (for example, user `HwHiAiUser` with ID `1001`), change the owner of the model directory and files in the directory to `1001` (this step can be omitted if using `root` privileges) and change the permission on the weight directory to `750`.

```shell
chown -R 1001:1001 {/path-to-weights/DeepSeek-V3.2}
chmod 750 {/path-to-weights/DeepSeek-V3.2}
```

---

## Installation

For details about how to install MindIE-LLM, see [Installation Guide](../../install/installing_MindIE.md).

---

## Inference Serving

### Atlas 800I A3 two-server deployment

#### Configuring the serving environment variables

Perform the following steps on both servers to set environment variables:

```shell
source /usr/local/lib/python3.11/site-packages/mindie_llm/set_env.sh # Set the environment variables required for running MindIE-LLM.
export MINDIE_LOG_TO_STDOUT=1                            # (Optional) Enable log output to the screen.
export TASK_QUEUE_ENABLE=0                               # Disable the task queue to avoid precision issues in multi-stream scenarios.
export HCCL_OP_EXPANSION_MODE="HOST"                     # Use the host mode to avoid occasional errors of communication operators.
export HCCL_BUFFSIZE=1050                                # Set the HCCL buffer size. You can refer to the typical configuration in the following sections.
# Memory optimization
export NPU_MEMORY_FRACTION=0.92                          # Set the NPU memory ratio.
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True" # Configure the expandable segments of the PyTorch memory.

# Performance optimization
export HCCL_ALGO="level0:NA;level1:pipeline"             # Set the HCCL algorithm.
export MINDIE_ASYNC_SCHEDULING_ENABLE=1                  # Enable asynchronous inference. (This step is optional but recommended. Peak performance optimization depends on asynchronous inference. The host and device latencies are mutually overlapped.)

# Environment variables related to multi-sever inference. Set them based on the actual environment.
export HCCL_CONNECT_TIMEOUT=7200                         # Set the HCCL connection timeout interval to prevent service startup failures caused by connection timeout.
export RANK_TABLE_FILE="/path/to/rank_table_file.json"   # Path to the rank_table_file.json file preconfigured in the environment setup section.
export MASTER_IP=xxx.xxx.xxx.xxx                         # IP address of the primary node
export MIES_CONTAINER_IP=xxx.xxx.xxx.xxx                 # IP address of the local host
export MASTER_PORT=xxxx                                  # Set the host port number. The value range is [0, 65535] and must not conflict with the port numbers of other services on the local host.
export GLOO_SOCKET_IFNAME=xxxx                           # Set the GLOO communication NIC.

# If a proxy is configured in the container, cancel the configuration to prevent two-server cluster communication exceptions.
unset http_proxy https_proxy
```

Note:

- The priority of `MIES_CONTAINER_IP` is higher than that of `ipAddress` in the configuration file. After the setting, the `MIES_CONTAINER_IP` of the primary node is used when a request is sent.
- Set `GLOO_SOCKET_IFNAME` to the NIC name corresponding to `MIES_CONTAINER_IP`. You can run the `ifconfig` command to view the NIC list and set this parameter to the corresponding NIC name. For details, see [FAQs](https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/faq/faq.md#gloo%E8%BF%9E%E6%8E%A5%E5%A4%B1%E8%B4%A5%E6%8A%A5%E9%94%99%EF%BC%9Aerror-failed-to-connect-errorso_error-connection-refused).

#### Setting serving parameters

Perform the following steps on both servers to modify the serving parameters. Go to the MindIE-LLM installation directory and edit the serving configuration file.

```shell
cd /usr/local/lib/python3.11/site-packages/mindie_llm
vim conf/config.json
```

Change the following parameters:

```json
{
    "ServerConfig" :
    {
        "httpsEnabled" : false, # After HTTPS is disabled, requests between the client and server are transmitted in plaintext. You are advised to disable HTTPS only on the secure intranet.
        ...
    },

    "BackendConfig" : {
        "npuDeviceIds" : [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]], # NPUs that can be used on the current host. For A3, each NPU has two devices, and eight NPUs have 16 devices in total.
        "multiNodesInferEnabled" : true,     # Enable multi-node inference.
        "interNodeTLSEnabled" : false,       # After cross-node TLS is disabled, communication data between nodes is transmitted in plaintext. You are advised to disable TLS only on the secure intranet.
        ...
        "ModelDeployConfig" :
        {
            ...
            "ModelConfig" : [
                {
                    ...
                    "modelName" : "DeepSeek-V3.2",                                      # Model name, which does not affect the startup of the serving service.
                    "modelWeightPath" : "/mnt/weights/DeepSeek-V3.2-w8a8-mtp-QuaRot",   # Weight path.
                    "worldSize" : 16,
                    "cpuMemSize" : 0,
                    "npuMemSize" : -1,
                    "backendType" : "torch",  # Select the backend of the inference framework. For DeepSeek-V3.2, torch must be selected.
                    "dp": 4,                  # Data parallelism
                    "tp": 8,                  # Tensor parallelism
                    "cp": 1,                  # Context parallelism
                    "sp": 1,                  # Sequence parallelism
                    "pp": 1,                  # Pipeline parallelism
                    "moe_ep": 32,             # MOE expert parallelism
                    "moe_tp": 1,              # MOE tensor parallelism
                    "plugin_params": "{\"plugin_type\":\"mtp\",\"num_speculative_tokens\":2}" # Enable the MTP feature and set the number of speculative tokens to 2.
                    ...
                }
            ]
        },
        ...
    },
    ...
}
```

`maxSeqLen`, `maxInputTokenLen`, `maxPrefillBatchSize`, `maxPrefillTokens`, `maxBatchSize`, and `maxIterTimes` need to be configured based on the actual scenario. You are advised to adjust them based on the model size and number of cards to achieve the optimal performance.

#### Starting the service

```shell
# Start the service. This step needs to be performed on both servers.
mindie_llm_server
```

After the command is executed, all parameters used for the startup are displayed. If the following information is displayed on both servers, the service is started successfully:

```shell
Daemon start success!
```

#### Verifying functions

After the service is started, open another terminal window and run the following command to send a request. Change `{MASTER_IP}` to `MASTER_IP` and `{PORT}` to the port number configured in the `config.json` serving configuration file.

```shell
curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{
  "model": "DeepSeek-V3.2",
  "max_tokens": 20,
  "messages": [{"role": "user", "content": "What is deep learning?"}]
}' http://{MASTER_IP}:{PORT}/v1/chat/completions
```

If the following information is displayed, the request is sent and the inference is successful:

```log
{"id":"endpoint_common_1","object":"chat.completion","created":1774785542,"model":"DeepSeek-V3.2","choices":[{"index":0,"message":{"role":"assistant","content":"Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to","tool_calls":[]},"logprobs":null,"finish_reason":"length"}],"usage":{"prompt_tokens":9,"prompt_tokens_details":{"cached_tokens":0},"completion_tokens":20,"completion_tokens_details":{"reasoning_tokens":0},"total_tokens":29}}
```

### Atlas 800I A2 Eight-Server MoE EP Deployment

Obtain the MoE EP initialization script by referring to [MindIE-Motor](https://gitcode.com/Ascend/MindIE-Motor/blob/v3.0.0/docs/en/user_guide/service_deployment/pd_separation_service_deployment.md#%E5%AE%89%E8%A3%85%E9%83%A8%E7%BD%B2).
The deployment directory structure is as follows:

```shell
boot_helper/boot.sh            # Pod execution script
collect_pd_cluster_logs.sh     # Log collection
conf/
    |_ mindie_env.json         # A2 environment variable configuration
    |_ mindie_env_a3.json      # A3 environment variable configuration
delete.sh                      # Pod deletion for stopping the service
deploy_ac_job.py               # Deployment startup. Generally, no modification is required.
deployment/                    # Kubernetes configuration. Generally, no modification is required.
user_config.json               # Service configuration for A2 hardware
user_config_base_A3.json       # Service configuration for A3 hardware
```

#### Configuring the serving environment variables

Modify `conf/mindie_env.json` (or `conf/mindie_env_a3.json` for A3).

```json
{
  "mindie_common_env": {
    ...
    "TASK_QUEUE_ENABLE": 0,            # Disable the task queue to avoid precision issues in multi-stream scenarios.
    "HCCL_BUFFSIZE": 1050,
    "HCCL_EXEC_TIMEOUT": 1200,         # Increase the value to avoid timeout. Note: In the earlier version of boot_helper/boot.sh, this variable is repeatedly defined. Increase the value accordingly.
    "MASTER_PORT": 10000               # Add this line. If the port is occupied, change the value to another free port.
  },
  "mindie_server_prefill_env": {
    ...
    "HCCL_OP_EXPANSION_MODE": "HOST",  # Use the host mode to avoid occasional errors of communication operators.
    "NPU_MEMORY_FRACTION": 0.8
  },
  "mindie_server_decode_env": {
    ...
    "TASK_QUEUE_ENABLE": 0,            # Disable the task queue to avoid precision issues in multi-stream scenarios.
    "NPU_MEMORY_FRACTION": 0.8,
    "HCCL_CONNECT_TIMEOUT": 7200,
    "HCCL_OP_EXPANSION_MODE": "HOST",  # Use the host mode to avoid occasional errors of communication operators.
    ...
  }
}
```

#### Setting serving parameters

Modify `user_config.json` (or `user_config_base_A3.json` for A3).

```json
{
  "deploy_config": {
    "p_instances_num": 1,
    "d_instances_num": 1,
    "single_p_instance_pod_num": 4,
    "single_d_instance_pod_num": 4,
    "p_pod_npu_num": 8,
    "d_pod_npu_num": 8,
    "image_name": "mindie:xxx",     # Set the image to be used.
    ...
  },
  "mindie_server_prefill_config": {
    ...
    "BackendConfig": {
      "npuDeviceIds": [
        [
          0,
          1,
          2,
          3,
          4,
          5,
          6,
          7
        ]
      ],
      "tokenizerProcessNumber": 1,
      "multiNodesInferEnabled": true,
      ...
      "ModelDeployConfig": {
        ...
        "ModelConfig": [
          {
            "modelInstanceType": "Standard",
            "modelName": "DeepSeek-V3.2",
            "modelWeightPath": "/path/to/DeepSeek-V3.2",               // Weight path. Change it to the actual path.
            "worldSize": 8,
            ...
            "backendType": "torch",
            ...
            "dp": 1,                                                   // The following lines configure dp, cp, and dp. You need to manually add them.
            "cp": 32,
            "tp": 1,
            "sp": 1,
            "moe_ep": 32,
            "pp": 1,
            "moe_tp": 1,
            "plugin_params": "{\"plugin_type\":\"mtp\",\"num_speculative_tokens\": 2}",  // Enable MTP and generate two tokens each time.
            ...
          }
        ]
      },
      ...
    }
  },
  "mindie_server_decode_config": {
    ...
    "BackendConfig": {
      ...
      "ModelDeployConfig": {
        ...
        "ModelConfig": [
          {
            ...
            "modelName": "DeepSeek-V3.2",
            "modelWeightPath": "/path/to/DeepSeek-V3.2",
            ...
            "backendType": "torch",
            "dp": 16,  # Manually add configurations such as dp.
            "cp": 1,
            "tp": 2,
            "sp": 1,
            "moe_ep": 32,
            "pp": 1,
            "moe_tp": 1,
            "plugin_params": "{\"plugin_type\":\"mtp\",\"num_speculative_tokens\": 2}"
          }
        ]
      },
      ...
    }
  }
}
```

#### Starting the service

Run the following commands in the Kubernetes environment:

```shell
# cd Deployment directory

# A2 environment
python3 deploy_ac_job.py
# Alternatively,
python3 deploy_ac_job.py --user_config_path user_config.json

# A3 environment
python3 deploy_ac_job.py --user_config_path user_config_base_A3.json
```

Logging:

```shell
bash collect_pd_cluster_logs.sh
```

It takes about **10 minutes** to start the service. Check the service status.

**Method 1**: Check logs.

```shell
bash collect_pd_cluster_logs.sh
# If "MindIE-MS coordinator is ready!" is displayed in the mindie-coordinator log, the service is started successfully.
```

**Method 2**: Check the pod status.

```shell
kubectl get pod -n mindie
# If the READY status of mindie-coordinator-master-0 is 1/1, the service is started successfully.
```

Output example:

```shell
bash-5.1# kubectl get pod -n mindie
NAME                          READY   STATUS    RESTARTS   AGE
mindie-controller-master-0    1/1     Running   0          23m
mindie-coordinator-master-0   1/1     Running   0          23m
mindie-server-d0-master-0     1/1     Running   0          23m
mindie-server-d0-worker-0     1/1     Running   0          23m
mindie-server-d0-worker-1     1/1     Running   0          23m
mindie-server-d0-worker-2     1/1     Running   0          23m
mindie-server-p0-master-0     1/1     Running   0          23m
mindie-server-p0-worker-0     1/1     Running   0          23m
mindie-server-p0-worker-1     1/1     Running   0          23m
mindie-server-p0-worker-2     1/1     Running   0          23m
```

#### Verifying functions

Single `curl` verification:

```shell
curl http://127.0.0.1:31015/v1/chat/completions -X POST -d '{
    "model": "DeepSeek-V3.2",
    "messages": [{ "role": "user", "content": "What is deep learning?" }],
    "stream": false,
    "temperature": 1.0,
    "max_tokens": 100
}'
```

Change the IP address and port number (`31015` by default) as required. The value of `model` must be the same as that of `modelName` specified in the configuration.

Response example:

```text
{"id":"endpoint_common_369","object":"chat.completion","created":1774838070,"model":"DeepSeek-V3.2","choices":[{"index":0,"message":{"role":"assistant","content":"Of course! Here is a comprehensive explanation of deep learning, broken down for clarity.\n\n### The Short Answer (The Elevator Pitch)\n\n**Deep learning is a subfield of machine learning that uses artificial neural networks with many layers (\"deep\" networks) to learn and make intelligent decisions from vast amounts of data.**\n\nThink of it as a way to automatically find complex patterns in data (like images, sound, or text) by passing it through a multi-layered processing system, where each layer extracts a","tool_calls":[]},"logprobs":null,"finish_reason":"length"}],"usage":{"prompt_tokens":9,"prompt_tokens_details":{"cached_tokens":0},"completion_tokens":100,"completion_tokens_details":{"reasoning_tokens":0},"total_tokens":109}}
```

#### Stopping the service

```shell
bash delete.sh
```

#### 128K context configuration

For 128K contexts, enable chunked prefill on the P node. The configuration of the A2 4+4 8-server MoE EP cluster is as follows:

```json
{
  "deploy_config": {
    "p_instances_num": 1,
    "d_instances_num": 1,
    "single_p_instance_pod_num": 4,
    "single_d_instance_pod_num": 4,
    "p_pod_npu_num": 8,
    "d_pod_npu_num": 8,
    ...
  "mindie_server_prefill_config": {
    ...
    "BackendConfig": {
      ...
      "ModelDeployConfig": {
        ...
        "ModelConfig": [
          {
            ...
            "dp": 4,
            "cp": 1,
            "tp": 8,
            "sp": 1,
            "moe_ep": 32,
            "pp": 1,
            "plugin_params": "{\"plugin_type\":\"mtp, splitfuse\",\"num_speculative_tokens\": 2}",   # Add the SplitFuse feature.
            "moe_tp": 1,
            ...
          }
        ]
      },
      "ScheduleConfig": {
        "templateType": "Mix",          # If this configuration does not exist, manually add this line.
        ...
        "maxPrefillBatchSize": 10,
        "maxPrefillTokens": 8192        # Number of tokens in each chunk.
      }
    }
  },
  "mindie_server_decode_config": {
    ...
    "BackendConfig": {
      ...
      "ModelDeployConfig": {
        "maxSeqLen": 128000,
        "maxInputTokenLen": 128000,
        "truncation": 0,
        "ModelConfig": [
          {
            ...
            "dp": 4,
            "cp": 1,
            "tp": 8,
            "sp": 1,
            "moe_ep": 32,
            "pp": 1,
            "moe_tp": 1,
            "plugin_params": "{\"plugin_type\":\"mtp\",\"num_speculative_tokens\": 2}",
            ...
          }
        ]
      },
      "ScheduleConfig": {
        ...
        "maxPrefillTokens": 128000,    # Required for verification. This value is not read in decoding.
        "maxBatchSize": 64,
        "maxIterTimes": 128000,
        ...
      }
    }
  }
}
```

## Accuracy Test

The following uses the GSM8K dataset as an example to describe how to perform an accuracy test.

**1. Obtaining Open-Source Datasets**

The following uses the GSM8K dataset as an example to describe how to use AISBench.

Obtain the open-source dataset by referring to [Dataset Preparation Guide](https://github.com/AISBench/benchmark/blob/master/docs/source_en/get_started/datasets.md), find the GSM8K dataset, download and decompress it, and save it to the datasets folder in the root path of the AISBench tool.

```shell
# Go to the AISBench installation directory. The default installation path `/usr/local/lib/python3.11/site-packages/ais_bench` is used as an example.
cd /usr/local/lib/python3.11/site-packages/ais_bench
# Go to the datasets directory.
cd datasets
# Copy the downloaded GSM8K dataset package (change `path/to/gsm8k.zip` to the actual path) to the current directory and decompress it.
cp path/to/gsm8k.zip ./
unzip gsm8k.zip
```

**2. Modifying the Configuration**

AISBench provides multiple configuration templates. This test mainly uses the following files in the `<aisbench_install_path>/benchmark/configs/models/vllm_api/` directory:

- `vllm_api_general_chat.py`: non-streaming inference configuration. Both non-streaming and streaming accuracy can be used.
- `vllm_api_stream_chat`: streaming inference configuration. Streaming is required for performance testing.
- `vllm_api_function_call_chat.py`: Function call is used when the BFCL dataset is tested.

The following uses `vllm_api_general_chat.py` as an example to describe the configuration:

```shell
# Go to the AISBench installation directory. The path may vary depending on the environment.
cd /usr/local/lib/python3.11/site-packages/ais_bench
# Edit the file.
vim benchmark/configs/models/vllm_api/vllm_api_general_chat.py
```

Example of the `vllm_api_general_chat.py` file:

```python
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="/path/to/DeepSeek-V3.2",  # Absolute path to the model serialized vocabulary file, which is generally the path to the model weight folder.
        model="DeepSeek-V3.2",          # Name of the model loaded on the server, which must be the same as the value of "modelName" in the configuration.
        request_rate=0,
        retry=2,
        host_ip="127.0.0.1",            # IP address of the inference service (IP address of the primary node in a multi-node inference scenario).
        host_port=1025,                 # Port of the inference service (port configured in the config.json file of the inference serving).
        max_out_len=4096,               # Maximum number of tokens output by the inference service
        batch_size=32,                  # Maximum number of concurrent requests to be sent, which is adjusted based on the server load.
        trust_remote_code=False,
        generation_kwargs=dict(
            # To set a specific parameter, cancel the corresponding comment.
            #top_p=0.95,
            #top_k=20,
            #seed=None,
            #temperature=1.0,
            #chat_template_kwargs={"enable_thinking": True},  # Whether to enable thinking.
        )
    )
]
```

**3. Running the following command to start the serving accuracy test**

```shell
ais_bench --models vllm_api_general_chat --datasets gsm8k_gen_4_shot_cot_chat_prompt --debug
```

The value of `--models` is the name of the configuration file used.

The command is executed successfully if the command output is as follows:

```shell
| dataset | version | metric | mode | vllm-api-stream-chat |
|----- | ----- | ----- | ----- | -----|
| gsm8k | e3c4be | accuracy | gen | 94.69 |
```

For more information about how to use AISBench, see [AISBench_benchmark](https://github.com/AISBench/benchmark).

The following table lists the accuracy test configurations of each dataset.

| Dataset Name  | Configuration Parameter                  | AISBench Request Command   | Accuracy Result        |
| ---          | ---                       | ---                | ---              |
| gsm8k        | max_out_len:4096          |  ais_bench --models vllm_api_general_chat --datasets gsm8k_gen_4_shot_cot_chat_prompt --debug                        | 94.69(94.69%) |
| ceval        | max_out_len:32000<br>enable_thinking:True     |  ais_bench --models vllm_api_general_chat --datasets  ceval_gen_0_shot_cot_chat_prompt --mode all                    | 92.1(92.1%)   |
| gpqa-diamond | max_out_len:32000<br>enable_thinking:True     | ais_bench --models vllm_api_general_chat --datasets  gpqa_gen_0_shot_cot_chat_prompt --mode all | 84.85(84.85%) |
| aime2024     | max_out_len:32000<br>enable_thinking:True     | ais_bench --models vllm_api_general_chat --datasets  aime2024_gen_0_shot_chat_prompt --mode all | 90.0(90.0%)   |
| bfcl-simple  | max_out_len:32000<br>temperature:0.001<br>enable_thinking:True     | ais_bench --models vllm_api_function_call_chat --datasets BFCL_gen_simple | 0.93(93%)     |

- Note: The accuracy result may fluctuate. You are advised to perform the measurement for multiple times.

## Performance Test

**1. Obtaining Open-Source Datasets**

During performance testing, you can construct GSM8K-formatted dataset based on the input and output lengths and the number of data records in the test, and then replace the generated dataset with `datasets/gsm8k/test.jsonl`.

**2. Configuring the `vllm_api_stream_chat.py` file (The following is an example.)**

```shell
# Go to the AISBench installation directory. The default installation path `/usr/local/lib/python3.11/site-packages/ais_bench` is used as an example.
cd /usr/local/lib/python3.11/site-packages/ais_bench
# Edit the vllm_api_stream_chat.py file. The following is a Python code example:
vim benchmark/configs/models/vllm_api/vllm_api_stream_chat.py
```

```python
from ais_bench.benchmark.models import VLLMCustomAPIChatStream

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="",                    # Absolute path to the model serialized vocabulary file, which is generally the path to the model weight folder.
        model="DeepSeek-V3.2",      # Name of the model loaded on the server. Set this parameter based on the name of the model pulled by the inference service. (If this parameter is set to an empty string, the model name is automatically obtained.)
        request_rate=0,             # Request sending frequency. One request is sent to the server every 1/request_rate second. If the value is less than 0.1, all requests are sent at a time.
        retry=2,
        host_ip="127.0.0.1",        # IP address of the inference service (IP address of the primary node in a multi-node inference scenario).
        host_port=1025,             # Port of the inference service (port configured in the config.json file of the inference serving).
        max_out_len=1024,           # Maximum number of tokens output by the inference service
        batch_size=16,              # Maximum number of concurrent requests to be sent
        generation_kwargs=dict(
            temperature=0,          # If the value is 0, postprocessing is disabled. You can set this parameter to another value as required.
            ignore_eos=True,        # To test the fixed-length output, set ignore_eos to True.
        )
    )
]
```

**3. Running the following command to start the serving performance test**

```shell
ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_0_shot_cot_str_perf --mode perf --summarizer default_perf --debug
```

The performance test is successful if the command output is as follows (example only):

```shell
╒═══════════════════════╤═══════╤══════════╤══════╤══════╤══════╤══════╤══════╤══════╤═══╕
│ Performance Parameters│ Stage │ Average  │ Min  │ Max  │Median│  P75 │  P90 │  P99 │ N │
╞═══════════════════════╪═══════╪══════════╪══════╪══════╪══════╪══════╪══════╪══════╪═══╡
│ E2EL                  │ total │ xxx ms   │ xxx  │ xxx  │ xxx  │ xxx  │ xxx  │ xxx  │ x │
├───────────────────────┼───────┼──────────┼──────┼──────┼──────┼──────┼──────┼──────┼───┤
│ TTFT                  │ total │ xxx ms   │ xxx  │ xxx  │ xxx  │ xxx  │ xxx  │ xxx  │ x │
├───────────────────────┼───────┼──────────┼──────┼──────┼──────┼──────┼──────┼──────┼───┤
│ TPOT                  │ total │ xxx ms   │ xxx  │ xxx  │ xxx  │ xxx  │ xxx  │ xxx  │ x │
├───────────────────────┼───────┼──────────┼──────┼──────┼──────┼──────┼──────┼──────┼───┤
│ ITL                   │ total │ xxx ms   │ xxx  │ xxx  │ xxx  │ xxx  │ xxx  │ xxx  │ x │
├───────────────────────┼───────┼──────────┼──────┼──────┼──────┼──────┼──────┼──────┼───┤
│ InputTokens           │ total │ xxxx     │ xxxx │ xxxx │ xxxx │ xxxx │ xxxx │ xxxx │ x │
├───────────────────────┼───────┼──────────┼──────┼──────┼──────┼──────┼──────┼──────┼───┤
│ OutputTokens          │ total │ xxxx     │ xxxx │ xxxx │ xxxx │ xxxx │ xxxx │ xxxx │ x │
├───────────────────────┼───────┼──────────┼──────┼──────┼──────┼──────┼──────┼──────┼───┤
│ OutputTokenThroughput │ total │ xxx tok/s│ xxx  │ xxx  │ xxx  │ xxx  │ xxx  │ xxx  │ x │
╘═══════════════════════╧═══════╧══════════╧══════╧══════╧══════╧══════╧══════╧══════╧═══╛
╒════════════════════════════╤═══════╤══════════╕
│ Common Metric              │ Stage │ Value    │
╞════════════════════════════╪═══════╪══════════╡
│ Benchmark Duration         │ total │ xxx ms   │
├────────────────────────────┼───────┼──────────┤
│ Total Requests             │ total │ xxx      │
├────────────────────────────┼───────┼──────────┤
│ Failed Requests            │ total │ xxx      │
├────────────────────────────┼───────┼──────────┤
│ Success Requests           │ total │ xxx      │
├────────────────────────────┼───────┼──────────┤
│ Concurrency                │ total │ xxx      │
├────────────────────────────┼───────┼──────────┤
│ Max Concurrency            │ total │ xxx      │
├────────────────────────────┼───────┼──────────┤
│ Request Throughput         │ total │ xxx req/s│
├────────────────────────────┼───────┼──────────┤
│ Total Input Tokens         │ total │ xxx      │
├────────────────────────────┼───────┼──────────┤
│ Prefill Token Throughput   │ total │ xxx tok/s│
├────────────────────────────┼───────┼──────────┤
│ Total generated tokens     │ total │ xxx      │
├────────────────────────────┼───────┼──────────┤
│ Input Token Throughput     │ total │ xxx tok/s│
├────────────────────────────┼───────┼──────────┤
│ Output Token Throughput    │ total │ xxx tok/s│
├────────────────────────────┼───────┼──────────┤
│ Total Token Throughput     │ total │ xxx tok/s│
╘════════════════════════════╧═══════╧══════════╛
```

**Key performance indicators:**

- `TTFT`: time to first token, the time from sending a request to receiving the first output token. It reflects the speed in the prefill phase.
- `TPOT`: time per output token, the average time required to generate a token in the decoding phase. It reflects the inference throughput capability.
- `Prefill Token Throughput`: prefill throughput, the number of tokens processed per second in the prefill phase. It reflects the efficiency of the first word response in long-context scenarios.
- `Output Token Throughput`: output throughput, that is, the number of output tokens generated per second in the decoding phase. It reflects the actual generation capability in the inference phase.

---

## Typical Configuration

The typical deployment modes of DeepSeek V3.2 are as follows:

| Deployment Configuration|Deployment Mode|  Machine Quantity| Card Quantity| Maximum Context Length| Parallelism Strategy| MTP Quantization<br>mtp=2 | chunked prefill | HCCL_BUFFSIZE (MB)|  NPU_MEM_FRACTION |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| A2 four-server cluster  | Prefill-decode co-location| 4 | 32  | 32K | MLA: DP4+TP8<br>MOE: EP32+TP1 | ✅ | ❌  | 512 | 0.8 |
| A2 four-server cluster  | Prefill-decode co-location| 4 | 32  | 64K | MLA: DP4+TP8<br>MOE: EP32+TP1 | ❌ | ✅  | 512 | 0.8 |
| A2 MoE EP | Prefill-decode disaggregation| 8 | 64  | 64K | P:<br>MLA: DP1+TP2+CP16<br>MOE: EP32+TP1<br>D:<br>MLA: DP8+TP4<br>MOE: EP32+TP1 | ✅ | ❌ | 1050 | 0.8 |
| A2 MoE EP| Prefill-decode disaggregation| 8 | 64  | 128K | P:<br>MLA: DP4+TP8<br>MOE: EP32+TP1<br>D:<br>MLA: DP4+TP8<br>MOE: EP32+TP1 | ✅ | P✅ D❌ | 1050 | 0.8 |
| A3 two-server cluster | Prefill-decode co-location| 2 | 16  | 32K | MLA: DP4+TP8<br>MOE: EP32+TP1 | ✅ | ❌ | 1050 | 0.92 |
| A3 two-server cluster | Prefill-decode co-location| 2 | 16  | 64K | MLA: DP4+TP8<br>MOE: EP32+TP1 | ❌ | ✅ | 1050 | 0.92 |
| A3 MoE EP| Prefill-decode disaggregation| 4 | 32  | 64K | P:<br>MLA: DP1+TP2+CP16<br>MOE: EP32+TP1<br>D:<br>MLA: DP8+TP4<br>MOE: EP32+TP1 | ✅ | ❌ | 1050 | 0.8 |
| A3 MoE EP| Prefill-decode disaggregation| 4 | 32  | 128K | P:<br>MLA: DP4+TP8<br>MOE: EP32+TP1<br>D:<br>MLA: DP4+TP8<br>MOE: EP32+TP1 | ✅ | P✅ D❌ | 1050 | 0.8 |

---

## Disclaimer

- The datasets and models referenced in this code repository are provided solely as illustrative examples for non-commercial purposes only. Users are solely responsible for complying with the respective licenses of these datasets and models. Huawei assumes no liability for any infringement disputes arising from your use of these datasets or models.
- If you encounter any issues while using the local code (including but not limited to functional defects or compliance concerns), please submit an issue in this code repository. We will review and address it promptly.
