# README

- [MiniCPM-V-2_6](https://github.com/OpenBMB/MiniCPM-V)是面向图文理解的端侧多模态大模型系列。该系列模型接受图像和文本输入，并提供高质量的文本输出.
- 此代码仓中实现了一套基于NPU硬件的MiniCPM-V推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

# 使用说明

## 特性矩阵
| 模型及参数量                     | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | 800I A2 BF16 | MindIE Service | 纯模型支持模态 | 服务化支持模态    |
|----------------------------|------------|---------------------------|------|------------|----------------|---------|------------|
| MiniCPM-V-2_6, 8B | 支持 TP 1、2、4、8      | 支持 TP 1、2           | √    | √          | √              | 文本、图片、视频   | 单轮对话/多轮对话 | 

## 路径变量解释

| 变量名               | 含义                                                                                                                                                             |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| working_dir       | 加速库及模型库下载后放置的目录                                                                                                                                                |
| llm_path          | 模型仓所在路径。若使用编译好的包，则路径为 `${working_dir}/MindIE-LLM/`；若使用gitcode下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models`                                          |
| script_path       | 脚本所在路径；minicpm_qwen2_v2的工作脚本所在路径为 `${llm_path}/examples/models/minicpm_qwen2_v2`                                                                                               |
| weight_path       | 模型权重路径                                                                                                                                                         |
| trust_remote_code  | 是否信任本地的可执行文件：默认不执行，传入此参数，则信任                                                                                                                |
| image_or_video_path        | 图片或视频所在文件夹的路径。当前图片仅支持".jpg", ".png", ".jpeg", ".bmp"四种格式。视频仅支持".mp4", ".wmv", ".avi"三种格式                                                                                                                              |
| max_batch_size    | 最大batch数                                                                                                                                                       |
| max_input_length  | 多模态模型的最大embedding长度。 |
| max_output_length | 生成的最大token数                                                                                                                                                    |




## 推理
**权重下载**

- [MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6/tree/main)

**模型文件拷贝**

将权重目录中的resampler.py拷贝到${llm_path}/examples/atb_models/atb_llm/models/minicpm_qwen2_v2目录下

**基础环境变量**
- 1.Toolkit, MindIE/ATB, ATB-SPEED等，参考[此README文件](../../../README.md)
- 2.Python其他第三方库依赖，参考[requirements_minicpm_qwen2_v2.txt](../../../requirements/models/requirements_minicpm_qwen2_v2.txt)
  ```shell
  pip install -r ${llm_path}/requirements/models/requirements_minicpm_qwen2_v2.txt
  ```

### 对话测试

- 运行启动脚本
  - 在\${llm_path}目录下执行以下命令
    ```shell
    bash ${script_path}/run_pa.sh --run --trust_remote_code ${weight_path} ${image_path}
    ```
- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0`

  - 以下环境变量与性能和内存优化相关，通常情况下无需修改
    ```shell
    export INF_NAN_MODE_ENABLE=0
    export ATB_OPERATION_EXECUTE_ASYNC=1
    export TASK_QUEUE_ENABLE=1
    export ATB_CONVERT_NCHW_TO_ND=1
    export ATB_CONTEXT_WORKSPACE_SIZE=0
    ```

## 服务化推理


- 打开配置文件

```shell
vim /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
```

- 更改配置文件

```json
{
...
"ServerConfig" :
{
...
"port" : 1040, #自定义
"managementPort" : 1041, #自定义
"metricsPort" : 1042, #自定义
...
"httpsEnabled" : false,
...
},

"BackendConfig": {
...
"npuDeviceIds" : [[0,1,2,3,4,5,6,7]],
...
"ModelDeployConfig":
{
"maxSeqLen" : 16384,
"maxInputTokenLen" : 16384,
"truncation" : false,
"ModelConfig" : [
{
"modelInstanceType": "Standard",
"modelName" : "minicpm_qwen2_v2", # 为了方便使用benchmark测试，modelname建议使用internvl
"modelWeightPath" : "/data_mm/weights/MiniCPM-V-2_6",
"worldSize" : 8,
...
"npuMemSize" : 1, #kvcache分配，可自行调整，单位是GB，切勿设置为-1，需要给vit预留显存空间。32GB机器建议设为1, 64GB机器可以设为8。
...
"trustRemoteCode" : false #默认为false，若设为true，则信任本地代码，用户需自行承担风险
}
]
},
"ScheduleConfig" :
{
...
"maxPrefillTokens" : 50000,
"maxIterTimes": 4096,
...
}
}
}
```

- 拉起服务化

```shell
cd /usr/local/Ascend/mindie/latest/mindie-service/bin
./mindieservice_daemon
```

- 另起一个新的容器端口，测试VLLM接口

```shell
curl 127.0.0.1:1040/generate -d '{
"prompt": [
{
"type": "image_url",
"image_url": ${图片路径}
},
{"type": "text", "text": "Explain the details in the image."}
],
"max_tokens": 512,
"stream": false,
"do_sample":true,
"repetition_penalty": 1.00,
"temperature": 0.01,
"top_p": 0.001,
"top_k": 1,
"model": "minicpm_qwen2_v2"
}'
```

- 另起一个新的容器端口，测试 OpenAI 接口

```shell
curl 127.0.0.1:1040/v1/chat/completions -d ' {
"model": "minicpm_qwen2_v2",
"messages": [{
"role": "user",
"content": [
{"type": "image_url", "image_url": ${图片路径}},
{"type": "text", "text": "Explain the details in the image."}
]
}],
"max_tokens": 512,
"do_sample": true,
"repetition_penalty": 1.00,
"temperature": 0.01,
"top_p": 0.001,
"top_k": 1
}'
```


##  benchmark精度测试方案
- 首先按照[服务化推理](#服务化推理)，拉起服务化

### TextVQA
- 另起一个新的容器端口，运行如下benchmark命令
```shell
# 输出benchmark运行日志
export MINDIE_LOG_TO_STDOUT="benchmark:1; client:1" 

benchmark \
--TestAccuracy True \
--DatasetPath ${textvqa_val.jsonl的绝对路径} \
--DatasetType textvqa  \
--ModelName ${与config.json中的modelName保持一致，建议为minicpm_qwen2_v2} \
--ModelPath ${模型权重的绝对路径} \
--TestType client \
--Concurrency 1 \
--TaskKind text \
--Tokenizer True \
--MaxOutputLen 20  \
--WarmupSize 1 \
--DoSampling False \
--TrustRemoteCode True \
--Http http://127.0.0.1:${端口号，与起服务化时的config.json保持一致} \
--ManagementHttp http://127.0.0.2:${端口号，与起服务化时的config.json保持一致} \
--SavePath ${日志输出路径}
```

### VideoBench
- 另起一个新的容器端口，运行如下benchmark命令
```shell
# 输出benchmark运行日志
export MINDIE_LOG_TO_STDOUT="benchmark:1; client:1" 

benchmark \
--TestAccuracy True \
--DatasetPath ${Eval_QA文件夹的绝对路径} \
--DatasetType videobench  \
--ModelName ${与config.json中的modelName保持一致，建议为minicpm_qwen2_v2} \
--ModelPath ${模型权重的绝对路径} \
--TestType client \
--Concurrency 1 \
--TaskKind text \
--Tokenizer True \
--MaxOutputLen 20  \
--WarmupSize 1 \
--DoSampling False \
--TrustRemoteCode True \
--Http http://127.0.0.1:${端口号，与起服务化时的config.json保持一致} \
--ManagementHttp http://127.0.0.2:${端口号，与起服务化时的config.json保持一致} \
--SavePath ${日志输出路径}
```

##  性能测试方案
- 另起一个新的容器端口，运行如下benchmark命令
```shell
# 输出benchmark运行日志
export MINDIE_LOG_TO_STDOUT="benchmark:1; client:1" 

benchmark \
--DatasetPath ${performance.jsonl的绝对路径} \
--DatasetType textvqa  \
--ModelName ${与config.json中的modelName保持一致，建议为minicpm_qwen2_v2} \
--ModelPath ${模型权重的绝对路径} \
--TestType client \
--Concurrency ${并发数，限制同时发起的连接数} \ # Client模式下，不超过endpoint所支持的最大连接数。取值范围：[1，1000]，默认值：128。
--RequestRate ${发送频率} \ # 指定一组发送频率，按照Distribution参数设置的模式进行发送，以每个频率完成一次推理，频率的取值范围：（0，10000]。
--TaskKind stream \
--Tokenizer True \
--MaxOutputLen 20  \
--WarmupSize 1 \
--DoSampling False \
--TrustRemoteCode True \
--Http http://127.0.0.1:${端口号，与起服务化时的config.json保持一致} \
--ManagementHttp http://127.0.0.2:${端口号，与起服务化时的config.json保持一致} \
--SavePath ${日志输出路径}
```