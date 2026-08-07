
# Kimi-K2

## 硬件要求

部署 Kimi-K2 模型用 W8A8 量化权重进行推理则至少需要 4 台 Atlas 800I A2 (8\*64G)。

## 权重

**权重下载**

### FP8 原始权重下载

- [Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct/tree/main)

### 权重转换（Convert FP8 weights to BF16）

NPU 侧权重转换

注意：

- Kimi-K2 模型基本复用 DeepSeek-V3 的模型结构，所以可以直接复用 DeepSeek-V2 的权重转换脚本。详见：[DeepSeek-V2 模型转换](https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/MindIE/LLM/DeepSeek/DeepSeek-V3/README.md#%E6%9D%83%E9%87%8D%E8%BD%AC%E6%8D%A2convert-fp8-weights-to-bf16)

### W8A8 量化权重生成（BF16 to INT8）

注意：

- Kimi-K2 模型基本复用 DeepSeek-V3 的模型结构，量化过程同理可以参考 [DeepSeek 模型量化方法介绍](https://gitcode.com/Ascend/msmodelslim/blob/26.1.0/example/DeepSeek/README.md)

## 推理前置准备

- 修改模型文件夹属组为 1001 -HwHiAiUser 属组（容器为 Root 权限可忽视）
- 执行权限为 750：

```sh
chown -R 1001:1001 {/path-to-weights/Kimi-K2-Instruct}
chmod -R 750 {/path-to-weights/Kimi-K2-Instruct}
```

多机场景下，需要配置 rank_table_file.json，具体方法请参考 [Rank Table File 配置指南](../../../../../docs/zh/user_guide/user_manual/rank_table_file_configuration.md)。

## 加载镜像

## 加载镜像

需要使用 mindie:2.2 及其后版本。

前往[昇腾社区/开发资源](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)下载适配，下载镜像前需要申请权限，耐心等待权限申请通过后，根据指南下载对应镜像文件。

完成之后，请使用 `docker images` 命令确认查找具体镜像名称与标签。

```bash
docker images
```

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
四机：
export WORLD_SIZE=32
export HCCL_EXEC_TIMEOUT=0
```

#### 第三方库安装

```bash
# Kimi-K2 有特定的 transformers 版本要求以及 blobfile 第三方库要求
pip install transformers==4.48.3
pip install blobfile
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
export RANK_TABLE_FILE={rank_table_file.json 路径}
```

workspace 内存分配算法选择，详见[加速库环境变量参考](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/acce/ascendtb/ascendtb_0032.html)

```bash
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
```

配置通信算法的编排展开位置

```bash
export HCCL_OP_EXPANSION_MODE="AIV"
```

HCCL 通信

```bash
export ATB_LLM_HCCL_ENABLE=1
export HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT=TRUE
```

异步发射

```bash
export MINDIE_ASYNC_SCHEDULING_ENABLE=1
```

绑核

```bash
export CPU_AFFINITY_CONF=1
```

队列优化特性

```bash
export TASK_QUEUE_ENABLE=2
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
"modelName" : "kimi_k2" # 不影响服务化拉起
"modelWeightPath" : "权重路径",
"worldSize":8,
```

Example：仅供参考，请根据实际情况修改

```json
{
    "Version" : "1.0.0",

    "ServerConfig" :
    {
        "ipAddress" : "改成主节点 IP",
        "managementIpAddress" : "改成主节点 IP",
        "port" : 1025,
        "managementPort" : 1026,
        "metricsPort" : 1027,
        "allowAllZeroIpListening" : false,
        "maxLinkNum" : 1000,
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
        "openAiSupport" : "vllm",
        "tokenTimeout": 3600,
        "e2eTimeout": 3600,
        "distDPServerEnabled":false
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
            "maxSeqLen" : 10240,
            "maxInputTokenLen" : 2048,
            "truncation" : false,
            "ModelConfig" : [
                {
                    "modelInstanceType" : "Standard",
                    "modelName" : "kimi_k2",
                    "modelWeightPath" : "/home/data/kimi-k2-w8a8",
                    "worldSize" : 8,
                    "cpuMemSize" : 0,
                    "npuMemSize" : -1,
                    "backendType" : "atb",
                    "trustRemoteCode" : true,
                    "dp": 8,
                    "tp": 4,
                    "moe_tp": 1,
                    "moe_ep": 32,
                    "models": {
                       "deepseekv2": {
                          "ep_level": 2
                       }
                    },
                    "async_scheduler_wait_time": 120,
                    "kv_trans_timeout": 10,
                    "kv_link_timeout": 1080
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

            "maxBatchSize" : 200,
            "maxIterTimes" : 8192,
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

### 精度化测试样例

需要开启确定性计算环境变量。

```bash
export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=true
export ATB_MATMUL_SHUFFLE_K_ENABLE=0
```

使用 [AISBench](https://github.com/AISBench/benchmark) 进行测试，具体方法可参考 [AISBench 资料](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/)

测试指令样例：

```shell
ais_bench --models vllm_api_stream_chat --datasets aime2024_gen_0_shot_chat_prompt --debug
```

### 常见问题

#### 服务化常见问题

1. 常见问题可参考 [DeepSeek](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/MindIE/LLM/DeepSeek/DeepSeek-V3#%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98)
2. 加载 tokenizer 报错
Kimi-K2 的权重和 tokenizer 加载方式是依赖与 Kimi 本身的代码加载，所以请确保打开 trust_remote_code 开关，打开此开关意味着 mindie 将调用本地的代码，您选择开启该开关意味着您认为本地代码的安全，如因运行本地代码产生的问题，华为不承担任何责任。

   ```json
   "ModelConfig" : [
      {
         ...
         "trustRemoteCode" : true,
         ...
      }
   ]
   ```

3. 报错显示 transformers 版本不一致
请确保 Kimi-K2 所需的第三方库版本正确

   ```bash
   pip install transformers==4.48.3
   pip install blobfile
   ```

4. 精度出现较大问题，回答内容乱码
请确保服务化配置参数的正确性

   ```json
   "ModelConfig" : [
      {
         ...
         "dp": 8,
         "tp": 4, # 确保 tp 的取值符合 world_size / tp >= 8
         ...
         "models": {
            "deepseekv2": {
               "ep_level": 2 # ep_level 需要设置为 2
            }
         },
         ...
      }
   ]
   ```

#### 权重路径权限问题

注意保证权重路径是可用的，执行以下命令修改权限，**注意是整个父级目录的权限**：

```shell
chown -R HwHiAiUser:HwHiAiUser {/path-to-weights}
chmod -R 750 {/path-to-weights}
```

#### 更多故障案例，请参考链接：<https://www.hiascend.com/document/caselibrary>
