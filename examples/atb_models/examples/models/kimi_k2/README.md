
# Kimi-K2

## 硬件要求
部署Kimi-K2模型用W8A8量化权重进行推理则至少需要4台Atlas 800I A2 (8\*64G)。

## 权重
**权重下载**
### FP8原始权重下载
- [Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct/tree/main)

### 权重转换（Convert FP8 weights to BF16）

NPU侧权重转换

注意：
- Kimi-K2模型基本复用DeepSeek-V3的模型结构，所以可以直接复用DeepSeek-V2的权重转换脚本。详见：[DeepSeek-V2模型转换](https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/MindIE/LLM/DeepSeek/DeepSeek-V3/README.md#%E6%9D%83%E9%87%8D%E8%BD%AC%E6%8D%A2convert-fp8-weights-to-bf16)

### W8A8量化权重生成(BF16 to INT8)

注意：
- Kimi-K2模型基本复用DeepSeek-V3的模型结构，量化过程同理可以参考 [DeepSeek模型量化方法介绍](https://gitee.com/ascend/msit/tree/br_noncom_MindStudio_8.0.0_POC_20251231/msmodelslim/example/DeepSeek)

## 推理前置准备

- 修改模型文件夹属组为1001 -HwHiAiUser属组（容器为Root权限可忽视）
- 执行权限为750：
```sh
chown -R 1001:1001 {/path-to-weights/Kimi-K2-Instruct}
chmod -R 750 {/path-to-weights/Kimi-K2-Instruct}
```

## 推理前置准备
- 检查机器网络情况
```
# 1.检查物理链接
for i in {0..7}; do hccn_tool -i $i -lldp -g | grep Ifname; done 
```
```
# 2.检查链接情况
for i in {0..7}; do hccn_tool -i $i -link -g ; done
```
```
# 3.检查网络健康情况
for i in {0..7}; do hccn_tool -i $i -net_health -g ; done
```
```
# 4.查看侦测ip的配置是否正确
for i in {0..7}; do hccn_tool -i $i -netdetect -g ; done
```
```
# 5.查看网关是否配置正确
for i in {0..7}; do hccn_tool -i $i -gateway -g ; done
```
```
# 6.检查NPU底层tls校验行为一致性，建议统一全部设置为0，避免hccl报错
for i in {0..7}; do hccn_tool -i $i -tls -g ; done | grep switch
```
```
# 7.NPU底层tls校验行为置0操作，建议统一全部设置为0，避免hccl报错
for i in {0..7};do hccn_tool -i $i -tls -s enable 0;done
```
- 获取每张卡的ip地址
```
for i in {0..7};do hccn_tool -i $i -ip -g; done
```
- 需要用户自行创建rank_table_file.json，参考如下格式配置

以下是一个双机用例，用户自行添加ip，补全device：
```
{
   "server_count": "2",
   "server_list": [
      {
         "device": [
            {
               "device_id": "0",
               "device_ip": "...",
               "rank_id": "0"
            },
            {
               "device_id": "1",
               "device_ip": "...",
               "rank_id": "1"
            },
            ...
            {
               "device_id": "7",
               "device_ip": "...",
               "rank_id": "7"
            },
         ],
         "server_id": "...",
         "container_ip": "..."
      },
      {
         "device": [
            {
               "device_id": "0",
               "device_ip": "...",
               "rank_id": "8"
            },
            {
               "device_id": "1",
               "device_ip": "...",
               "rank_id": "9"
            },
            ...
            {
               "device_id": "7",
               "device_ip": "...",
               "rank_id": "15"
            },
         ],
         "server_id": "...",
         "container_ip": "..."
      },
   ],
   "status": "completed",
   "version": "1.0"
}
```
| 参数          |  说明                                                       |
|---------------|------------------------------------------------------------|
|  server_count |  总节点数                                                   |
|  server_list  |  server_list中第一个server为主节点                           |
|  device_id    |  当前卡的本机编号，取值范围[0, 本机卡数)                       |
|  device_ip    |  当前卡的ip地址，可通过hccn_tool命令获取                       |
|  rank_id      |  当前卡的全局编号，取值范围[0, 总卡数)                         |
|  server_id    |  当前节点的ip地址                                             |
|  container_ip |  容器ip地址（服务化部署时需要），若无特殊配置，则与server_id相同 |

rank_table_file.json配置完成后，需要执行命令修改权限为640
```sh
chmod -R 640 {rank_table_file.json路径}
```

## 加载镜像
需要使用mindie:2.2及其后版本。

前往[昇腾社区/开发资源](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)下载适配，下载镜像前需要申请权限，耐心等待权限申请通过后，根据指南下载对应镜像文件。

完成之后，请使用`docker images`命令确认查找具体镜像名称与标签。
```
docker images
```

## 容器启动

#### 启动容器

- 执行以下命令启动容器（参考）：
```sh
docker run -itd --privileged  --name= {容器名称}  --net=host \
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
   -v  {/权重路径:/权重路径}  \
   -v  {/rank_table_file.json路径:/rank_table_file.json路径}  \
    {swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:1.0.0-XXX-800I-A2-arm64-py3.11（根据加载的镜像名称修改）}  \
   bash
```
#### 进入容器

- 执行以下命令进入容器（参考）：
```sh
docker exec -it {容器名称} bash
```

#### 设置基础环境变量
```
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/atb-models/set_env.sh
source /usr/local/Ascend/mindie/set_env.sh
```
#### 开启通信环境变量
```
export ATB_LLM_HCCL_ENABLE=1
export ATB_LLM_COMM_BACKEND="hccl"
export HCCL_CONNECT_TIMEOUT=7200 # 该环境变量需要配置为整数，取值范围[120,7200]，单位s
四机：
export WORLD_SIZE=32
export HCCL_EXEC_TIMEOUT=0
```

#### 第三方库安装
```
# Kimi-K2有特定的transformers版本要求以及blobfile第三方库要求
pip install transformers==4.48.3
pip install blobfile
```

## 纯模型推理
【使用场景】使用相同输入长度和相同输出长度，构造多Batch去测试纯模型性能

#### 参数配置
- Kimi-K2 需要配置ep_level为2
```
cd /usr/local/Ascend/atb-models/atb_llm/
vim conf/config.json

修改如下 ep_level 参数:

"deepseekv2": {
    "eplb": {
        ...
    },
    "ep_level": 2,
    ...
}
```
- tp并行参数需要做一定限制：总卡数 / tp >= 8。例如：总卡数为32，则tp需要小于等于4，满足 32 / 4 >= 8。


#### 精度测试
- 进入modeltest路径
```
cd /usr/local/Ascend/atb-models/tests/modeltest/
```
- 运行测试脚本

Step1.主副节点分别先清理残余进程：
```
pkill -9 -f 'mindie|python'
```
Step2.需在所有机器上同时执行：
```
bash run.sh pa_[data_type] [dataset] ([shots]) [batch_size] [model_name] ([is_chat_model]) [weight_dir] [rank_table_file] [world_size] [node_num] [rank_id_start] [master_address] ([parallel_params]) 
```
参数说明：
1. `data_type`：为数据类型，根据权重目录下config.json的data_type选择bf16或者fp16，例如：pa_bf16。
2. `dataset`：可选full_BoolQ、full_CEval等，相关数据集可至[魔乐社区MindIE](https://modelers.cn/MindIE)下载，（下载之前，需要申请加入组织，下载之后拷贝到/usr/local/Ascend/atb-models/tests/modeltest/路径下）CEval与MMLU等数据集需要设置`shots`（通常设为5）。
3. `batch_size`：为`batch数`。
4. `model_name`：为`deepseekv3`。
5. `is_chat_model`：为`是否支持对话模式，若传入此参数，则进入对话模式`。
6. `weight_dir`：为模型权重路径。
7. `rank_table_file`：为“前置准备”中配置的`rank_table_file.json`路径。
8. `world_size`：为总卡数。
9. `node_num`：为当前节点编号，即`rank_table_file.json`的`server_list`中顺序确定。
10. `rank_id_start`：为当前节点起始卡号，即`rank_table_file.json`中当前节点第一张卡的`rank_id`，Atlas 800I-A2双机场景下，主节点为0，副节点为8。
11. `master_address`：为主节点ip地址，即`rank_table_file.json`的`server_list`中第一个节点的ip。
12. `parallel_params`: 接受一组输入，格式为[dp,tp,sp,moe_tp,moe_ep,pp,microbatch_size,cp]，如[8,1,-1,1,8,-1,-1,-1]

测试脚本运行如下，以四机为例：

样例 AIME2024

主节点
```
bash run.sh pa_bf16 full_AIME2024 10 deepseekv3 {/path/to/weights/Kimi-K2} trust_remote_code {/path/to/xxx/ranktable.json} 32 4 0 {主节点IP} [8, 4, -1, 1, 32, -1, -1, -1]
# 0 代表从0号卡开始推理，之后的机器依次从8，16，24。
```
副节点1
```
bash run.sh pa_bf16 full_AIME2024 10 deepseekv3 {/path/to/weights/Kimi-K2} trust_remote_code {/path/to/xxx/ranktable.json} 32 4 8 {主节点IP} [8, 4, -1, 1, 32, -1, -1, -1]
# 0 代表从0号卡开始推理，之后的机器依次从8，16，24。
```
副节点2
```
bash run.sh pa_bf16 full_AIME2024 10 deepseekv3 {/path/to/weights/Kimi-K2} trust_remote_code {/path/to/xxx/ranktable.json} 32 4 16 {主节点IP} [8, 4, -1, 1, 32, -1, -1, -1]
# 0 代表从0号卡开始推理，之后的机器依次从8，16，24。
```
副节点3
```
bash run.sh pa_bf16 full_AIME2024 10 deepseekv3 {/path/to/weights/Kimi-K2} trust_remote_code {/path/to/xxx/ranktable.json} 32 4 24 {主节点IP} [8, 4, -1, 1, 32, -1, -1, -1]
# 0 代表从0号卡开始推理，之后的机器依次从8，16，24。
```

#### 性能测试
- 进入modeltest路径：
```
cd /usr/local/Ascend/atb-models/tests/modeltest/
```
Step1.主副节点分别先清理残余进程：
```
pkill -9 -f 'mindie|python'
```
Step2.需在所有机器上同时执行：
```
bash run.sh pa_[data_type] performance [case_pair] [batch_size] ([prefill_batch_size]) [model_name] ([is_chat_model]) [weight_dir] [rank_table_file] [world_size] [node_num] [rank_id_start] [master_address] ([parallel_params]) 
```
参数说明：
1. `data_type`：为数据类型，根据权重目录下config.json的data_type选择bf16或者fp16，例如：pa_bf16。
2. `case_pair`：[最大输入长度,最大输出长度]。
3. `batch_size`：为`batch数`。
4. `prefill_batch_size`：为可选参数，设置后会固定prefill的batch size。
5. `model_name`：为`deepseekv3`。
6. `is_chat_model`：为`是否支持对话模式，若传入此参数，则进入对话模式`。
7. `weight_dir`：为模型权重路径。
8. `rank_table_file`：为“前置准备”中配置的`rank_table_file.json`路径。
9. `world_size`：为总卡数。
10. `node_num`：为当前节点编号，即`rank_table_file.json`的`server_list`中顺序确定。
11. `rank_id_start`：为当前节点起始卡号，即`rank_table_file.json`中当前节点第一张卡的`rank_id`，Atlas 800I-A2双机场景下，主节点为0，副节点为8。
12. `master_address`：为主节点ip地址，即`rank_table_file.json`的`server_list`中第一个节点的ip。
13. `parallel_params`: 接受一组输入，格式为[dp,tp,sp,moe_tp,moe_ep,pp,microbatch_size,cp]，如[8,1,-1,1,8,-1,-1,-1]

测试脚本运行如下，以四机为例：

主节点
```
bash run.sh pa_bf16 performance [[256,256]] 1 deepseekv3 {/path/to/weights/Kimi-K2} trust_remote_code {/path/to/xxx/ranktable.json} 32 4 0 {主节点IP}
# 0 代表从0号卡开始推理，之后的机器依次从8，16，24。
```
副节点1
```
bash run.sh pa_bf16 performance [[256,256]] 1 deepseekv3 {/path/to/weights/Kimi-K2} trust_remote_code {/path/to/xxx/ranktable.json} 32 4 8 {主节点IP}
# 0 代表从0号卡开始推理，之后的机器依次从8，16，24。
```
副节点2
```
bash run.sh pa_bf16 performance [[256,256]] 1 deepseekv3 {/path/to/weights/Kimi-K2} trust_remote_code {/path/to/xxx/ranktable.json} 32 4 16 {主节点IP}
# 0 代表从0号卡开始推理，之后的机器依次从8，16，24。
```
副节点3
```
bash run.sh pa_bf16 performance [[256,256]] 1 deepseekv3 {/path/to/weights/Kimi-K2} trust_remote_code {/path/to/xxx/ranktable.json} 32 4 24 {主节点IP}
# 0 代表从0号卡开始推理，之后的机器依次从8，16，24。
```

## 服务化推理
【使用场景】对标真实客户上线场景，使用不同并发、不同发送频率、不同输入长度和输出长度分布，去测试服务化性能
#### 配置服务化环境变量

变量含义：expandable_segments-使能内存池扩展段功能，即虚拟内存特性。更多详情请查看[昇腾环境变量参考](https://www.hiascend.com/document/detail/zh/Pytorch/600/apiref/Envvariables/Envir_009.html)。
```
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```
服务化需要`rank_table_file.json`中配置`container_ip`字段。
所有机器的配置应该保持一致，除了环境变量的MIES_CONTAINER_IP为本机ip地址。
```
export MIES_CONTAINER_IP={容器ip地址}
export RANK_TABLE_FILE={rank_table_file.json路径}
```
workspace内存分配算法选择，详见[加速库环境变量参考](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/acce/ascendtb/ascendtb_0032.html)
```
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
```
配置通信算法的编排展开位置
```
export HCCL_OP_EXPANSION_MODE="AIV"
```
HCCL通信
```
export ATB_LLM_HCCL_ENABLE=1
export HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT=TRUE
```
异步发射
```
export MINDIE_ASYNC_SCHEDULING_ENABLE=1
```
绑核
```
export CPU_AFFINITY_CONF=1
```
队列优化特性
```
export TASK_QUEUE_ENABLE=2
```

#### 修改服务化参数
```
cd /usr/local/Ascend/mindie/latest/mindie-service/
vim conf/config.json
```
修改以下参数
```
"httpsEnabled" : false, # 如果网络环境不安全，不开启HTTPS通信，即“httpsEnabled”=“false”时，会存在较高的网络安全风险
...
"multiNodesInferEnabled" : true, # 开启多机推理
...
# 若不需要安全认证，则将以下两个参数设为false
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
```
{
    "Version" : "1.0.0",

    "ServerConfig" :
    {
        "ipAddress" : "改成主节点IP",
        "managementIpAddress" : "改成主节点IP",
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
```
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
```
Daemon start success!
```
则认为服务成功启动。

#### 另起客户端
进入相同容器，向服务端发送请求。

更多信息可参考官网信息：[MindIE Service](https://www.hiascend.com/document/detail/zh/mindie/100/mindieservice/servicedev/mindie_service0285.html)。

### 精度化测试样例

需要开启确定性计算环境变量。
```
export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=true
export ATB_MATMUL_SHUFFLE_K_ENABLE=0
```
使用aisbench进行测试，具体方法可参考[aisbench资料](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/)

测试指令样例：
```
ais_bench --models vllm_api_stream_chat --datasets aime2024_gen_0_shot_chat_prompt --debug
```

### 常见问题

#### 服务化常见问题
1. 常见问题可参考[DeepSeek](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/MindIE/LLM/DeepSeek/DeepSeek-V3#%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98)
2. 加载tokenizer报错
Kimi-K2的权重和tokenizer加载方式是依赖与Kimi本身的代码加载，所以请确保打开trust_remote_code开关，打开此开关意味着mindie将调用本地的代码，您选择开启该开关意味着您认为本地代码的安全，如因运行本地代码产生的问题，华为不承担任何责任。
```
"ModelConfig" : [
   {
      ...
      "trustRemoteCode" : true,
      ...
   }
]
```
3. 报错显示transformers版本不一致
请确保Kimi-K2所需的第三方库版本正确
```
pip install transformers==4.48.3
pip install blobfile
```
4. 精度出现较大问题，回答内容乱码
请确保服务化配置参数的正确性
```
"ModelConfig" : [
    {
        ...
        "dp": 8,
        "tp": 4, # 确保tp的取值符合 world_size / tp >= 8
        ...
        "models": {
           "deepseekv2": {
              "ep_level": 2 # ep_level 需要设置为2
           }
        },
        ...
    }
]
```
#### 权重路径权限问题
注意保证权重路径是可用的，执行以下命令修改权限，**注意是整个父级目录的权限**：
```sh
chown -R HwHiAiUser:HwHiAiUser {/path-to-weights}
chmod -R 750 {/path-to-weights}
```
#### 更多故障案例，请参考链接：https://www.hiascend.com/document/caselibrary
