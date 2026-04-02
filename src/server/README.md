# MindIE_Server

## 简介
推理服务端，提供模型服务化能力。
面向通用模型的推理服务化场景，实现开放、可拓展的推理服务化平台架构，支持对接业界主流推理框架接口，满足大语言模型、文生图等多类型模型的高性能推理需求。

## 组件
MindIE_Server主要包含如下组件：
| 组件 |  简介 |
| ----   | -----    |
| daemon(服务启动) | 负责推理服务启动，加载配置文件，初始化其他模块 |
| config_manager(配置管理) | 负责服务化相关配置的解析和校验 |
| endpoint(北向接口) | 面向推理服务开发者提供极简易用的API接口，支持Triton/OpenAI/TGI/vLLM第三方主流推理框架请求接口|
| gmis(统一API) | 支持推理流程的工作流定义扩展，以工作流为驱动，实现从推理任务调度到任务执行的可扩展架构，适应各类推理方法如投机推理、LoRA推理、LLMA、Prompt增强等的快速落地|
| tokenzier | 自研tokenzier |

## 配置参数说明
配置文件参数说明
|  配置项   | 取值类型  |    取值范围  |  配置说明  |
|  ----  | ----  | ----  |  ----  |
| Version  | std::string | "1.0.0"  |  标注配置文件版本，当前版本指定为1.0.0，不支持修改。 |
| LogConfig  | map | -  | 日志相关配置。|
| ModelDeployConfig  | map | -  | 模型部署相关配置。|
| ServerConfig  | map | -  | 服务端相关配置，例如ip:port、网络请求、网络安全等。|
| BackendConfig  | map | - |  模型后端相关配置，包含调度、模型相关配置。 |

详情请参见[MindIE昇腾社区用户文档](https://www.hiascend.com/document/detail/zh/mindie "访问MindIE昇腾社区")。

### 提供的RESTful API列表如下：
#### 表1 服务状态查询API（管理面的查询类接口）
| API名称 | 接口类型 | URL | 说明 | 支持框架 |
| ---- | ---- | ---- | ---- | ---- |
|Server Live | GET | /v2/health/live | 检查服务器是否在线。  | 	Triton |
| Server Ready |  GET| 	/v2/health/ready |  检查服务器是否准备就绪。 | Triton |
| Model Ready | GET | /v2/models/\${MODEL_NAME}[/versions/${MODEL_VERSION}]/ready | 检查模型是否准备就绪。  | Triton |
| health |  GET| /health | 服务健康检查。  | TGI/vLLM |
| 查询TGI EndPoint信息 | GET | 	/info | 查询TGI EndPoint信息。  |	TGI  |
| Slot统计 | GET | /v2/models/\${MODEL_NAME}[/versions/${MODEL_VERSION}]/getSlotCount |  参考Triton格式，自定义的Slot统计信息查询接口。 | 原生 |
| 健康探针接口 | GET | /health/timed[-${TIMEOUT}] | 检查推理流程是否正常。  | 原生 |
| 优雅退出接口 | GET | 	/stopService | 实现整个服务的优雅退出。调用该接口时，会等待服务中正在执行和等待的所有请求完成，并关闭服务，等待时所有推理接口将不可用。  | 原生 |
| 静态配置采集接口 | GET | /v1/config | 采集静态配置。  | 原生 |
|动态状态采集接口  | GET | 	/v1/status |  采集动态状态。 | 原生 |
| 指定实例身份接口 | POST |  /v1/role/${role}| 	指定实例身份。  | 	原生 |
| 服务指标接口（JSON格式） | GET | 	/metrics-json |  获取推理服务过程中请求的TTFT（Time To First Token）、TBT（Time Between Tokens）的动态平均值（默认近1000个请求的平均值），正在执行请求数、正在等待请求数量、剩余NPUblock数量。 | 原生 |
| 服务监控指标查询接口（普罗格式） | GET | 	/metrics | 	查询推理服务化的相关服务监控指标。  | 原生 |

#### 表2 模型/服务查询API（业务面的查询接口）
| API名称 | 接口类型 | URL | 说明 | 支持框架 |
| ---- | ---- | ---- | ---- | ---- |
| models列表 | GET | /v1/models |  	列举当前可用模型列表。 |  	OpenAI|
| model详情 | GET | /v1/models/{model} | 	查询模型信息。  | OpenAI |
| 服务元数据查询 | GET | /v2 |  获取服务元数据。 | Triton |
| 查询模型配置 | GET | /v2/models/\${MODEL_NAME}[/versions/${MODEL_VERSION}]/config |  查询模型配置。 | 	Triton |
#### 表3 推理API（业务面的业务接口）
| API名称 | 接口类型 | URL | 说明 | 支持框架 |
| ---- | ---- | ---- | ---- | ---- |
| 推理任务 | POST | / | TGI推理接口，stream==false返回文本推理结果，stream==true返回流式推理结果。  | TGI |
| 推理任务 | POST | /generate | TGI和vLLM的推理接口，通过请求参数来区分是哪种服务的接口。  | TGI/vLLM |
| 推理任务 | POST | /generate_stream |  TGI流式推理接口，使用Server-Sent Events格式返回结果。 | TGI |
| 推理任务 | POST | /v1/chat/completions | OpenAI文本推理接口。  | OpenAI |
| 推理任务 | POST | /infer |  原生推理接口，支持文本/流式返回结果。 | 原生 |
| 推理任务 |  POST| /infer_token |  原生推理接口，实现token输入的文本/流式推理。 | 原生 |
| 推理任务 | POST |	/v2/models/\${MODEL_NAME}[/versions/${MODEL_VERSION}]/infer  | Triton的token推理接口。  | Triton |
| 推理任务 | POST | /v2/models/\${MODEL_NAME}[/versions/${MODEL_VERSION}]/stopInfer | 参考Triton接口定义，提供提前终止请求接口。  | 原生 |
| 推理任务 | POST | /v2/models/\${MODEL_NAME}[/versions/${MODEL_VERSION}]/generate |  Triton文本推理接口。 | Triton |
| 推理任务 | POST | /v2/models/\${MODEL_NAME}[/versions/${MODEL_VERSION}]/generate_stream | Triton流式推理接口。  |Triton  |
|推理任务  | POST | /v1/tokenizer |  计算token数量。 | 原生 |
| 推理任务 | GET| /dresult |  调度器与D实例间，存在一个长连接，D实例每推理出一个结果，就通过该长连接响应给调度器。 | PD分离相关 |

## 使用简介
以Linux curl工具发送请求，以兼容的Triton RESTful接口为例，详尽的使用说明可以查看[MindIE昇腾社区用户文档](https://www.hiascend.com/document/detail/zh/mindie "访问MindIE昇腾社区")：
### 健康检查
检查服务状态是否正常

```shell    
# 请求1
GET https://{ip}:{port}/v2/health/live 
# 响应1
# 正常场景：服务状态正常 不会返回结果

# 异常场景：从节点异常
{
    "message": no contact node detected
    "no_contact_node": ["node(10.10.10.10) is no contact"]
}

# 请求2
GET https://{ip}:{port}/v2/models/llama_65b/ready 
# 响应1
# 正常场景：状态码200，无内容

# 异常场景：其他状态码
```
### 查询服务元数据
查询服务元数据信息，例如max_iter_times、max_prefill_batch_size等

请求样例:
```shell
GET https://{ip}:{port}/v2
```
响应样例, 状态码200:
```json
{
    "name": "MindeIE Server",
    "version": "{version}",
    "extensions": {
        "max_iter_times": 512,
        "prefill_policy_type": 0,
        "decode_policy_type": 0,
        "max_prefill_batch_size": 50,
        "max_prefill_tokens": 8192
    }

}
```
### 文本推理接口
提供文本推理处理功能

请求样例:
```shell
POST https://{ip}:{port}/v2/models/llama_65b/generate
```
请求体：

单模态文本模型
```json
{
    "id":"a123",
    "text_input": "My name is Olivier and I",
    "parameters": {
        "details": true,
        "do_sample": true,
        "max_new_tokens": 20,
        "repetition_penalty": 1.1,
        "seed": 123,
        "temperature": 1,
        "top_k": 10,
        "top_p": 0.99,
        "batch_size": 100,
        "typical_p": 0.5,
        "watermark": false,
        "priority": 5,
        "timeout": 10
    }
}
```
多模态文本模型
```json
{
    "id":"a123",
    "text_input": [
        {"type": "text", "text": "My name is Olivier and I"},
        {
            "type": "image_url",
            "image_url": "/xxxx/test.png"
        }
    ],
    "parameters": {
        "details": true,
        "do_sample": true,
        "max_new_tokens":20,
        "repetition_penalty": 1.1,
        "seed": 123,
        "temperature": 1,
        "top_k": 10,
        "top_p": 0.99,
        "batch_size":100,
        "typical_p": 0.5,
        "watermark": false,
        "priority": 5,
        "timeout": 10
    }
}
```
响应样例：
```json
{
    "id" "a123",
    "model_name": "llama_65b",
    "model_version": null,
    "text_output": "am living in South ...",
    "details": {
        "finish_reason": "eos_token",
        "generated_tokens": 221,
        "first_token_cost": null,
        "decode_cost": null
    }
}
```




了解更多信息请访问[MindIE昇腾社区](https://www.hiascend.com/software/mindie "访问MindIE昇腾社区")

