# 使用指导 

## 场景说明

Server提供EndPoint模块对推理服务化协议和接口封装，兼容Triton/OpenAI/TGI/vLLM等第三方框架接口。使用单节点安装模式安装Server之后，用户使用客户端（Linux curl命令，Postman工具等）发送HTTP/HTTPS请求，即可调用EndPoint提供的接口。

>[!NOTE]说明 
>HTTP协议存在安全风险，建议您使用HTTPS安全协议。

## EndPoint RESTful接口使用说明

HTTP/HTTPS请求的URL的IP地址和端口号在config.json中进行配置，详情请参见[ServerConfig参数说明](../user_manual/service_parameter_configuration.md#serverconfig参数说明)。

- 以Linux curl工具发送generate请求，URL请求格式如下：
    - 操作类型：**POST**
    - **URL：http_\[__s\]_://**_\{ip\}:\{port\}_**/generate**

- 未开启HTTPS，发送推理请求：

    ```bash
    curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{
      "inputs": "My name is Olivier and I",
      "parameters": {
        "details": true,
        "do_sample": true,
        "repetition_penalty": 1.1,
        "return_full_text": false,
        "seed": null,
        "temperature": 1,
        "top_p": 0.99
      }
    }' http://{ip}:{port}/generate
    ```

- HTTPS双向认证的请求方式示例： 

    ```bash
    curl --location --request POST 'https://{ip}:{port}/generate' \
    --header 'Content-Type: application/json' \
    --cacert /home/runs/static_conf/ca/ca.pem \
    --cert /home/runs/static_conf/cert/client.pem \
    --key /home/runs/static_conf/cert/client.key.pem \
    --data-raw '{
        "inputs": "My name is Olivier and I",
        "parameters": {
            "best_of": 1,
            "decoder_input_details": false,
            "details": false,
            "do_sample": true,
            "max_new_tokens": 20,
            "repetition_penalty": 2,
            "return_full_text": false,
            "seed": 12,
            "temperature": 0.1,
            "top_k": 1,
            "top_p": 0.9,
            "truncate": 1024
        }
    }'
    ```

    >[!NOTE]说明 
    >- --cacert：验签证书文件路径。
    >- ca.pem：Server服务端证书的验签证书/根证书。
    >- --cert：客户端证书文件路径。
    >- client.pem：客户端证书。
    >- --key：客户端私钥文件路径。
    >- client.key.pem：客户端证书私钥（未加密，建议采用加密密钥）。
    >请用户根据实际情况对相应参数进行修改。

提供的RESTful API列表如下：

**表 1**  服务状态查询API（内部接口的查询类接口）

|API|接口类型|URL|说明|支持框架|
|--|--|--|--|--|
|Server Live|GET|/v2/health/live|检查服务器是否在线。|Triton|
|Server Ready|GET|/v2/health/ready|检查服务器是否准备就绪。|Triton|
|Model Ready|GET|/v2/models/${MODEL_NAME}[/versions/\${MODEL_VERSION}]/ready|检查模型是否准备就绪。|Triton|
|health|GET|/health|服务健康检查。|TGIvLLM|
|查询TGI EndPoint信息|GET|/info|查询TGI EndPoint信息。|TGI|
|Slot统计|GET|/v2/models/${MODEL_NAME}[/versions/\${MODEL_VERSION}]/getSlotCount|参考Triton格式，自定义的Slot统计信息查询接口。|原生|
|健康探针接口|GET|/health/timed[-$\{TIMEOUT\}]|检查推理流程是否正常。|原生|
|优雅退出接口|GET|/stopService|实现整个服务的优雅退出。调用该接口时，会等待服务中正在执行和等待的所有请求完成，并关闭服务，等待时所有推理接口将不可用。|原生|
|静态配置采集接口|GET|/v1/config|采集静态配置。|原生|
|动态状态采集接口|GET|/v1/status|采集动态状态。|原生|
|指定实例身份接口|POST|/v1/role/${role}|指定实例身份。|原生|
|动态状态采集接口|GET|/v2/status|采集动态状态。|原生|
|指定实例身份接口|POST|/v2/role/$\{role\}|指定实例身份。|原生|
|服务指标接口（JSON格式）|GET|/metrics-json|获取推理服务过程中请求的TTFT（Time To First Token）、TBT（Time Between Tokens）的动态平均值（默认近1000个请求的平均值），正在执行请求数、正在等待请求数量、剩余NPUblock数量。|原生|
|服务管控指标查询接口（普罗格式）|GET|/metrics|查询推理服务化的相关服务管控指标|原生|
|动态加载lora接口|POST|/v1/load_lora_adapter|动态加载lora|OpenAI|
|动态卸载lora接口|POST|/v1/unload_lora_adapter|动态卸载lora|OpenAI|

**表 2**  模型/服务查询API（业务面的查询接口）

|API|接口类型|URL|说明|支持框架|
|--|--|--|--|--|
|models列表|GET|/v1/models|列举当前可用模型列表。|OpenAI|
|model详情|GET|/v1/models/{model}|查询模型信息。|OpenAI|
|服务元数据查询|GET|/v2|获取服务元数据。|Triton|
|模型元数据查询|GET|/v2/models/${MODEL_NAME}[/versions/\${MODEL_VERSION}]|查询模型元数据信息。|Triton|
|查询模型配置|GET|/v2/models/${MODEL_NAME}[/versions/\${MODEL_VERSION}]/config|查询模型配置。|Triton|

**表 3**  推理API（业务面的业务接口）

|API|接口类型|URL|说明|支持框架|
|--|--|--|--|--|
|推理任务|POST|/|TGI推理接口，stream==false返回文本推理结果，stream==true返回流式推理结果。|TGI|
|POST|/generate|TGI和vLLM的推理接口，通过请求参数来区分是哪种服务的接口。|TGIvLLM|
|POST|/generate_stream|TGI流式推理接口，使用Server-Sent Events格式返回结果。|TGI|
|POST|/v1/chat/completions|OpenAI文本/流式推理接口。|OpenAI|
|POST|/v1/completions|vLLM兼容OpenAI文本/流式推理接口。|OpenAI|
|POST|/infer|原生推理接口，支持文本/流式返回结果。|原生|
|POST|/infer_token|原生推理接口，实现token输入的文本/流式推理。|原生|
|POST|/v2/models/${MODEL_NAME}[/versions/\${MODEL_VERSION}]/infer|Triton的token推理接口。|Triton|
|POST|/v2/models/${MODEL_NAME}[/versions/\${MODEL_VERSION}]/stopInfer|参考Triton接口定义，提供提前终止请求接口。|原生|
|POST|/v2/models/${MODEL_NAME}[/versions/\${MODEL_VERSION}]/generate|Triton文本推理接口。|Triton|
|POST|/v2/models/${MODEL_NAME}[/versions/\${MODEL_VERSION}]/generate_stream|Triton流式推理接口。|Triton|
|POST|/v1/tokenizer|计算token数量。|原生|
|GET|/dresult|调度器与D实例间，存在一个长连接，D实例每推理出一个结果，就通过该长连接响应给调度器。|PD分离相关|

>[!NOTE]说明 
>
>- $\{MODEL\_NAME\}字段指定需要查询的模型名称。
>- \[/versions/$\{MODEL\_VERSION\}\]字段暂不支持，不传递。
