# DeepSeek-V3

## 硬件要求

部署 DeepSeek-V3 模型用 W8A8 量化权重进行推理则至少需要 2 台 Atlas 800I A2 (8\*64G)。

## 权重

**权重下载**

### FP8 原始权重下载

FP8 权重可从以下来源下载：

| 模型 | HuggingFace | ModelScope |
|------|-------------|------------|
| DeepSeek-V3 | [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3/tree/main) | [ModelScope](https://modelscope.cn/models/deepseek-ai/DeepSeek-V3) |
| DeepSeek-V3-Base | [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/tree/main) | 暂无 |
| DeepSeek-V3-0324 | [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/tree/main) | [ModelScope](https://modelscope.cn/models/deepseek-ai/DeepSeek-V3-0324) |

> 下载后请自行放入权重目录，后续推理步骤需引用该路径。

### 权重转换（Convert FP8 weights to BF16）

NPU 侧权重转换

注意：

- DeepSeek 官方没有针对 DeepSeek-V3 提供新的权重转换脚本，所以复用 DeepSeek-V2 的权重转换脚本。
- 若用户使用上方脚本下载权重，则无需使用以下 git clone 命令，直接进入权重转换脚本目录。

```sh
git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
```

```sh
cd ModelZoo-PyTorch/MindIE/LLM/DeepSeek/DeepSeek-V2/NPU_inference
```

```sh
python fp8_cast_bf16.py --input-fp8-hf-path {/path/to/DeepSeek-V3} --output-bf16-hf-path {/path/to/DeepSeek-V3-bf16}
```

目前 npu 转换脚本不会自动复制 tokenizer 等文件，需要将原始权重的 tokenizer.json, tokenizer_config.json 等文件复制到转换之后的路径下。

注意：

- `/path/to/DeepSeek-V3` 表示 DeepSeek-V3 原始权重路径，`/path/to/DeepSeek-V3-bf16` 表示权重转换后的新权重路径。
- 由于模型权重较大，请确保您的磁盘有足够的空间放下所有权重，例如 DeepSeek-V3 在转换前权重约为 640G 左右，在转换后权重约为 1.3T 左右。
- 推理作业时，也请确保您的设备有足够的空间加载模型权重，并为推理计算预留空间。

### BF16 原始权重下载

也可以通过 HuggingFace，ModelScope 等开源社区直接下载 BF16 模型权重：

|  来源 |  链接 |
|---|---|
| huggingface  |  <https://huggingface.co/unsloth/DeepSeek-V3-bf16/> |
| modelscope  | <https://modelscope.cn/models/unsloth/deepseek-V3-bf16/>  |
| modelers | <https://modelers.cn/models/State_Cloud/DeepSeek-V3-BF16>  |

DeepSeek-V3-0324 的 BF16 模型权重链接为：

|  来源 |  链接 |
|---|---|
| huggingface  |  <https://huggingface.co/unsloth/DeepSeek-V3-0324-BF16/> |
| modelscope  | <https://www.modelscope.cn/models/unsloth/DeepSeek-V3-0324-BF16/>  |

### W8A8 量化权重生成(BF16 to INT8)

目前支持：生成模型 W8A8 混合量化权重，使用 histogram 量化方式（MLA：W8A8 量化，MOE：W8A8 dynamic pertoken 量化）。

详情请参考 [DeepSeek 量化说明](https://gitcode.com/Ascend/msmodelslim/blob/master/example/DeepSeek/README.md)

注意：DeepSeek-V3 模型权重较大，量化权重生成时间较久，请耐心等待；具体时间与校准数据集大小成正比，10 条数据大概需花费 3 小时。

### 昇腾原生量化 W8A8 权重下载(动态量化)

也可以通过 Modelers 等开源社区直接下载昇腾原生量化 W8A8 模型权重：

- [DeepSeek-R1](https://modelers.cn/models/State_Cloud/Deepseek-R1-bf16-hfd-w8a8)
- [DeepSeek-V3-0324](https://modelers.cn/models/Modelers_Park/DeepSeek-V3-0324-w8a8)

## 推理前置准备

- 修改权重目录下的 config.json 文件

注意：在本仓实现中，DeepSeek-V3 目前沿用 DeepSeek_v3 代码框架。

多机场景下，需要配置 rank_table_file.json，具体方法请参考 [Rank Table File 配置指南](../../../../../docs/zh/user_guide/user_manual/rank_table_file_configuration.md)。

- 修改模型文件夹属组为 1001 -HwHiAiUser 属组（容器为 Root 权限可忽视），执行权限为 750：

```sh
chown -R 1001:1001 {/path-to-weights/DeepSeek-V3}
chmod -R 750 {/path-to-weights/DeepSeek-V3}
```

## 加载镜像

需要使用 mindie:2.0.T3 及其后版本。

前往[昇腾社区/开发资源](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)下载适配，下载镜像前需要申请权限，耐心等待权限申请通过后，根据指南下载对应镜像文件。

DeepSeek-V3 的镜像版本：2.0.T3-800I-A2-py311-openeuler24.03-lts
镜像加载后的名称：swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:2.0.T3-800I-A2-py311-openeuler24.03-lts。

完成之后，请使用 `docker images` 命令确认查找具体镜像名称与标签。

```bash
docker images
```

各组件版本配套如下：

| 组件 | 版本 |
| - | - |
| MindIE | 2.0.T3 |
| CANN | 8.0.T63 |
| Pytorch | 6.0.T700 |
| MindStudio | Msit: br_noncom_MindStudio_8.0.0_POC_20251231 分支 |
| Ascend HDK | 24.1.0 |

## 容器启动

### 启动容器

- 执行以下命令启动容器（参考）：

   ```sh
   docker run -itd --privileged --name= {容器名称} --net=host \
      --shm-size 500g \
      --device=/dev/davinci0 \
      --device=/dev/davinci1 \
      --device=/dev/davinci2 \
      --device=/dev/davinci3 \
      --device=/dev/davinci4 \
      --device=/dev/davinci5 \
      --device=/dev/davinci6 \
      --device=/dev/davinci7 \
      --device=/dev/davinci_manager \
      --device=/dev/hisi_hdc \
      --device /dev/devmm_svm \
      -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
      -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
      -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
      -v /usr/local/sbin:/usr/local/sbin \
      -v /etc/hccn.conf:/etc/hccn.conf \
      -v {/权重路径:/权重路径} \
      -v {/rank_table_file.json 路径:/rank_table_file.json 路径} \
      {swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:1.0.0-XXX-800I-A2-arm64-py3.11（根据加载的镜像名称修改）} \
      bash
   ```

#### 进入容器

- 执行以下命令进入容器（参考）：

   ```sh
   docker exec -it {容器名称} bash
   ```

#### 设置基础环境变量

   ```bash
   source /usr/local/Ascend/cann/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   source /usr/local/Ascend/atb-models/set_env.sh
   source /usr/local/Ascend/mindie/set_env.sh
   ```

#### 开启通信环境变量

   ```bash
   export ATB_LLM_HCCL_ENABLE=1
   export ATB_LLM_COMM_BACKEND="hccl"
   export HCCL_CONNECT_TIMEOUT=7200 # 该环境变量需要配置为整数，取值范围[120,7200]，单位 s
   双机：
   export WORLD_SIZE=16
   四机：
   export WORLD_SIZE=32
   export HCCL_EXEC_TIMEOUT=0
   ```

## 服务化推理

【使用场景】对标真实客户上线场景，使用不同并发、不同发送频率、不同输入长度和输出长度分布，去测试服务化性能

### 配置服务化环境变量

变量含义：expandable_segments-使能内存池扩展段功能，即虚拟内存特性。更多详情请查看[昇腾环境变量参考](https://www.hiascend.com/document/detail/zh/Pytorch/600/apiref/Envvariables/Envir_009.html)。

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

服务化需要 `rank_table_file.json` 中配置 `container_ip` 字段。
所有机器的配置应该保持一致，除了环境变量的 MIES_CONTAINER_IP 为本机 ip 地址。

```bash
export MIES_CONTAINER_IP={容器 ip 地址}
export RANKTABLEFILE={rank_table_file.json 路径}
```

#### 修改服务化参数

```bash
cd /usr/local/Ascend/mindie/latest/mindie-service/
vim conf/config.json
```

修改以下参数

```json
"httpsEnabled" : false, # 如果网络环境不安全，不开启 HTTPS 通信，即“httpsEnabled”=“false”时，会存在较高的网络安全风险
...
"multiNodesInferEnabled" : true, # 开启多机推理
...
# 若不需要安全认证，则将以下两个参数设为 false
"interCommTLSEnabled" : false,
"interNodeTLSEnabled" : false,
...
"npudeviceIds" : [[0,1,2,3,4,5,6,7]],
...
"modelName" : "DeepSeek-V3" # 不影响服务化拉起
"modelWeightPath" : "权重路径",
"worldSize":8,
```

Example：仅供参考，请根据实际情况修改

```json
{
    "Version" : "1.0.0",
    "LogConfig" :
    {
        "logLevel" : "Info",
        "logFileSize" : 20,
        "logFileNum" : 20,
        "logPath" : "logs/mindie-server.log"
    },

    "ServerConfig" :
    {
        "ipAddress" : "改成主节点 IP",
        "managementIpAddress" : "改成主节点 IP",
        "port" : 1025,
        "managementPort" : 1026,
        "metricsPort" : 1027,
        "allowAllZeroIpListening" : false,
        "maxLinkNum" : 1000, //如果是 4 机，建议 300
        "httpsEnabled" : false,
        "fullTextEnabled" : false,
        "tlsCaPath" : "security/ca/",
        "tlsCaFile" : ["ca.pem"],
        "tlsCert" : "security/certs/server.pem",
        "tlsPk" : "security/keys/server.key.pem",
        "tlsCrlPath" : "security/certs/",
        "tlsCrlFiles" : ["server_crl.pem"],
        "managementTlsCaFile" : ["management_ca.pem"],
        "managementTlsCert" : "security/certs/management/server.pem",
        "managementTlsPk" : "security/keys/management/server.key.pem",
        "managementTlsCrlPath" : "security/management/certs/",
        "managementTlsCrlFiles" : ["server_crl.pem"],
        "metricsTlsCaFile" : ["metrics_ca.pem"],
        "metricsTlsCert" : "security/certs/metrics/server.pem",
        "metricsTlsPk" : "security/keys/metrics/server.key.pem",
        "metricsTlsCrlPath" : "security/metrics/certs/",
        "metricsTlsCrlFiles" : ["server_crl.pem"],
        "inferMode" : "standard",
        "interCommTLSEnabled" : false,
        "interCommPort" : 1121,
        "interCommTlsCaPath" : "security/grpc/ca/",
        "interCommTlsCaFiles" : ["ca.pem"],
        "interCommTlsCert" : "security/grpc/certs/server.pem",
        "interCommPk" : "security/grpc/keys/server.key.pem",
        "interCommTlsCrlPath" : "security/grpc/certs/",
        "interCommTlsCrlFiles" : ["server_crl.pem"],
        "openAiSupport" : "vllm"
    },

    "BackendConfig" : {
        "backendName" : "mindieservice_llm_engine",
        "modelInstanceNumber" : 1,
        "npuDeviceIds" : [[0,1,2,3,4,5,6,7]],
        "tokenizerProcessNumber" : 8,
        "multiNodesInferEnabled" : true,
        "multiNodesInferPort" : 1120,
        "interNodeTLSEnabled" : false,
        "interNodeTlsCaPath" : "security/grpc/ca/",
        "interNodeTlsCaFiles" : ["ca.pem"],
        "interNodeTlsCert" : "security/grpc/certs/server.pem",
        "interNodeTlsPk" : "security/grpc/keys/server.key.pem",
        "interNodeTlsCrlPath" : "security/grpc/certs/",
        "interNodeTlsCrlFiles" : ["server_crl.pem"],
        "ModelDeployConfig" :
        {
            "maxSeqLen" : 10000,
            "maxInputTokenLen" : 2048,
            "truncation" : true,
            "ModelConfig" : [
                {
                    "modelInstanceType" : "Standard",
                    "modelName" : "DeepSeek-V3",
                    "modelWeightPath" : "/home/data/dsv3_base_step178000",
                    "worldSize" : 8,
                    "cpuMemSize" : 5,
                    "npuMemSize" : -1,
                    "backendType" : "atb",
                    "trustRemoteCode" : false
                }
            ]
        },

        "ScheduleConfig" :
        {
            "templateType" : "Standard",
            "templateName" : "Standard_LLM",
            "cacheBlockSize" : 128,

            "maxPrefillBatchSize" : 8,
            "maxPrefillTokens" : 2048,
            "prefillTimeMsPerReq" : 150,
            "prefillPolicyType" : 0,

            "decodeTimeMsPerReq" : 50,
            "decodePolicyType" : 0,

            "maxBatchSize" : 8,
            "maxIterTimes" : 1024,
            "maxPreemptCount" : 0,
            "supportSelectBatch" : false,
            "maxQueueDelayMicroseconds" : 5000,
            "maxFirstTokenWaitTime": 2500
        }
    }
}
```

#### 拉起服务化

```bash
# 以下命令需在所有机器上同时执行
# 解决权重加载过慢问题
export OMP_NUM_THREADS=1
# 设置显存比
export NPU_MEMORY_FRACTION=0.95
# 拉起服务化
cd /usr/local/Ascend/mindie/latest/mindie-service/
./bin/mindieservice_daemon
```

执行命令后，首先会打印本次启动所用的所有参数，然后直到出现以下输出：

```text
Daemon start success!
```

则认为服务成功启动。

#### 另起客户端

进入相同容器，向服务端发送请求。

更多信息可参考官网信息：[MindIE Service](https://www.hiascend.com/document/detail/zh/mindie/100/mindieservice/servicedev/mindie_service0285.html)。

### 精度测试

精度测试使用 [AISBench](https://github.com/AISBench/benchmark) 工具。

**1、数据集准备**

请参考[数据集准备指南](https://github.com/AISBench/benchmark/blob/master/docs/source_zh_cn/get_started/datasets.md)获取开源数据集，下载解压后放置于 AISBench 工具根路径的 `datasets` 文件夹下。

以 MMLU 数据集为例：

```shell
cd /usr/local/lib/python3.11/site-packages/ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mmlu.zip
unzip mmlu.zip
rm mmlu.zip
```

**2、修改配置**

以 `<aisbench_install_path>/benchmark/configs/models/vllm_api/vllm_api_general_chat.py` 为例，配置内容示例如下：

```python
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="/path/to/DeepSeek-V3", # 模型权重文件夹路径
        model="DeepSeek-V3", # 与服务化 config.json 中的 modelName 一致
        request_rate=0,
        retry=2,
        host_ip="127.0.0.1", # 推理服务 IP（多机场景为主节点 IP）
        host_port=1025, # 服务化 config.json 中配置的 port
        max_out_len=20, # MMLU 为选择题，设为 20 即可
        batch_size=1, # 精度测试需设为 1
        trust_remote_code=False,
    )
]
```

**3、执行测试**

```shell
ais_bench --models vllm_api_general_chat --datasets mmlu_gen_5_shot_str --mode all
```

> 注意：MMLU 数据集中有约为 1.4w 条数据，推理耗时会比较长。`max_out_len` 设为 20 即可（选择题只需输出选项字母）。

### 性能调优示例

以下将以 800I-A3 单机部署 W8A8 权重为例，介绍通用的性能调优手段

#### 混合并行

``` json
{
   "ModelConfig" : [
      {
         "dp": 2,
         "tp": 8,
         "moe_tp": 4,
         "moe_ep": 4
      }
   ]
}
```

若要使用动态 EP，当前仅支持 `moe_tp = 1`

``` json
{
   "ModelConfig" : [
      {
         "dp": 2,
         "tp": 8,
         "moe_tp": 1,
         "moe_ep": 16,
         "models": {
            "deepseekv2": {
               "ep_level": 2
            }
         }
      },
   ]
}
```

#### MTP

MTP 特性对于较低并发性能提升更为明显，当前版本推荐 MTP=1 或 2

``` json
{
   "ModelConfig": [
      {
         "plugin_params": "{\"plugin_type\":\"mtp\",\"num_speculative_tokens\": 1}"
      }
   ]
}
```

#### NZ 格式 KV Cache

``` json
{
   "ModelConfig": [
      {
         "models": {
            "deepseekv2": {
               "kv_cache_options": {
                  "enable_nz": true
               }
            }
         }
      }
   ]
}
```

#### 权重预取

``` json
{
   "ModelConfig": [
      {
         "models": {
            "deepseekv2": {
               "enable_oproj_prefetch": true,
               "enable_mlapo_prefetch": true
            }
         }
      }
   ]
}
```

### 常见问题

#### 服务化常见问题

1. 若出现 out of memory 报错，可适当调高 NPU_MEMORY_FRACTION 环境变量（默认值为 0.8），适当调低服务化配置文件 config.json 中 maxSeqLen、maxInputTokenLen、maxPrefillBatchSize、maxPrefillTokens、maxBatchSize 等参数。

    ```bash
    export NPU_MEMORY_FRACTION=0.96
    ```

2. 若出现 hccl 通信超时报错，可配置以下环境变量。

    ```bash
    export HCCL_CONNECT_TIMEOUT=7200 # 该环境变量需要配置为整数，取值范围[120,7200]，单位 s
    export HCCL_EXEC_TIMEOUT=0
    ```

3. 若出现 AttributeError：'IbisTokenizer' object has no attribute 'cache_path'。

   Step1: 进入环境终端后执行。

      ```bash
      pip show mies_tokenizer
      ```

      默认出现类似如下结果，重点查看 `Location`

      ```text
      Name: mies_tokenizer
      Version: 0.0.1
      Summary: ibis tokenizer
      Home-page:
      Author:
      Author-email:
      License:
      Location: /usr/local/python3.10.13/lib/python3.10/site-packages
      Requires:
      Required-by:
      ```

   Step2: 打开 `Location` 路径下的./mies_tokenizer/tokenizer.py 文件。

      ```bash
      vim /usr/local/python3.10.13/lib/python3.10/site-packages/mies_tokenizer/tokenizer.py
      ```

   Step3: 对以下两个函数代码进行修改。

      ```diff
      def __del__(self):
      - dir_path = file_utils.standardize_path(self.cache_path)
      + cache_path = getattr(self, 'cache_path', None)
      + if cache_path is None:
      + return
      + dir_path = file_utils.standardize_path(cache_path)
            file_utils.check_path_permission(dir_path)
            all_request = os.listdir(dir_path)
      ```

      以及

      ```diff
      def _get_cache_base_path(child_dir_name):
            dir_path = os.getenv("LOCAL_CACHE_DIR", None)
            if dir_path is None:
               dir_path = os.path.expanduser("~/mindie/cache")
      - if not os.path.exists(dir_path):
      - os.makedirs(dir_path)
      + os.makedirs(dir_path, exist_ok=True)
               os.chmod(dir_path, 0o750)
      ```

4. 若出现 `UnicodeEncodeError: 'ascii' codec can't encode character`\uff5c`in position 301:ordinal not in range(128)`。

   这是因为由于系统在写入或打印日志 ASCII 编码 deepseek 的词表失败，导致报错，不影响服务化正常运行。如果需要规避，需要/usr/local/Ascend/atb-models/atb_llm/runner/model_runner.py 的第 145 行注释掉：print_log(rank, logger.info, f'init tokenizer done: {self.tokenizer}')。

5. 从节点无法和主节点建立 rpc 通信

   若出现多级部署从节点无法和主节点建立 rpc 通信问题，子节点报 RPC 问题，可能原因：防火墙拦截，排查方法：使用指令查看防火墙状态，如果开启防火墙，每台机器都需要关闭防火墙；

   查看防火墙状态：

      ```bash
      sudo systemctl status firewalld
      ```

   临时关闭防火墙，该操作存在安全隐患，请谨慎操作，该命令适用于 linux 系统，其它系统需要根据实际情况修改：

      ```bash
      sudo systemctl stop firewalld
      ```

6. 无进程内存残留

   如果卡上有内存残留，且有进程，可以尝试以下指令：

      ```bash
      pkill -9 -f 'mind|python'
      ```

   如果卡上有内存残留，但无进程，可以尝试以下指令：

      ```bash
      npu-smi set -t reset -i 0 -c 0 #重启 npu 卡
      npu-smi info -t health -i <card_idx> -c 0 #查询 npu 告警
      ```

      例：

      ```bash
      npu-smi set -t reset -i 0 -c 0 #重启 npu 卡 0
      npu-smi info -t health -i 2 -c 0 #查询 npu 卡 2 告警
      ```

   如果卡上有进程残留，无进程，且重启 NPU 卡无法消除残留内存，请尝试 reboot 重启机器

7. 日志收集

   遇到推理报错时，请打开日志环境变量，收集日志信息。

      - 算子库日志|默认输出路径为"~/atb/log"

         ```bash
         export ASDOPS_LOG_LEVEL = INFO
         export ASDOPS_LOG_TO_FILE = 1
         ```

      - 加速库日志|默认输出路径为"~/mindie/log/debug"

         ```bash
         export MINDIE_LOG_LEVEL = INFO
         export MINDIE_LOG_TO_FILE = 1
         ```

      - MindIE Service 日志|默认输出路径为"~/mindie/log/debug"

         ```bash
         export MINDIE_LOG_TO_FILE = 1
         export MINDIE_LOG_TO_LEVEL = debug
         ```

      - CANN 日志收集|默认输出路径为"~/ascend"

         ```bash
         export ASCEND_GLOBAL_LOG_TO_LEVEL = 1
         ```

8. 多机无法拉起 DeepSeek-R1 模型推理，HCCL 报错

      ```bash
      # 检查 NPU 底层 tls 校验行为一致性，建议统一全部设置为 0，避免 hccl 报错
      for i in {0..7}; do hccn_tool -i $i -tls -g ; done | grep switch
      ```

      ```bash
      # NPU 底层 tls 校验行为置 0 操作，建议统一全部设置为 0，避免 hccl 报错
      for i in {0..7};do hccn_tool -i $i -tls -s enable 0;done
      ```

#### 权重路径权限问题

注意保证权重路径是可用的，执行以下命令修改权限，**注意是整个父级目录的权限**：

```sh
chown -R HwHiAiUser:HwHiAiUser {/path-to-weights}
chmod -R 750 {/path-to-weights}
```

#### 更多故障案例，请参考链接：<https://www.hiascend.com/document/caselibrary>
