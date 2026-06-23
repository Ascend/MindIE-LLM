# Usage Guidance

## Overview

The Server provides EndPoint to encapsulate inference serving protocols and APIs. It is compatible with third-party framework APIs such as Triton, OpenAI, TGI, and vLLM. After the Server is installed in single-server mode, you can use a client (Linux curl command, Postman tool, and etc.) to send HTTP/HTTPS requests to call APIs provided by EndPoint.

>[!NOTE]NOTE
>HTTPS is recommended, as it is more secure than HTTP.

## Description of EndPoint RESTful APIs

The IP address and port number of an HTTP/HTTPS request URL are configured in the `config.json` file. For details, see [ServerConfig parameters](./service_parameter_configuration.md#parameters-in-serverconfig).

- URL format of a generate request sent by Linux curl:
    - Operation type: POST
    - URL: `http[s]://{ip}:{port}/generate`

- Inference request sent with HTTPS disabled:

    ```bash
    curl -H "Accept: application/json" -H "Content-Type: application/json" -X POST -d '{
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

- Request sent with HTTPS bidirectional authentication enabled:

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

    >[!NOTE]NOTE
    >- --`cacert`: path to the signature verification certificate file.
    >- `ca.pem`: signature verification certificate or root certificate of the Server certificate.
    >- --`cert`: path to the client certificate file.
    >- `client.pem`: client certificate.
    >- --`key`: path to the client private key file.
    >- `client.key.pem`: private key of the client certificate. (The private key is not encrypted. You are advised to use an encrypted key.)
    >Change the parameters as needed.

The following table lists the provided RESTful APIs.

**Table 1** Service status query APIs (internal query APIs)

|API|Interface Type|URL|Description|Framework|
|--|--|--|--|--|
|Server Live|GET|/v2/health/live|Checks whether the server is online.|Triton|
|Server Ready|GET|/v2/health/ready|Checks whether the server is ready.|Triton|
|Model Ready|GET|/v2/models/${MODEL_NAME}[/versions/\${MODEL_VERSION}]/ready|Checks whether the model is ready.|Triton|
|health|GET|/health|Performs service health checks.|TGIvLLM|
|TGI Endpoint information query|GET|/info|Queries the TGI EndPoint information.|TGI|
|Slot statistics|GET|/v2/models/${MODEL_NAME}[/versions/\${MODEL_VERSION}]/getSlotCount|Queries customized slot statistics based on the Triton format.|Native|
|Health probe|GET|/health/timed[-$\{TIMEOUT\}]|Checks whether an inference process is normal.|Native|
|Graceful exit|GET|/stopService|Implements graceful exit of the entire service. When this API is called, the system stops a service until all requests that are being executed and waiting are complete. During the waiting, all inference APIs are unavailable.|Native|
|Collecting static configurations|GET|/v1/config|Collects static configurations.|Native|
|Collecting dynamic status|GET|/v1/status|Collects dynamic status.|Native|
|Specifying an instance role|POST|/v1/role/${role}|Specifies an instance role.|Native|
|Collecting dynamic status|GET|/v2/status|Collects dynamic status.|Native|
|Specifying an instance role|POST|/v2/role/$\{role\}|Specifies an instance role.|Native|
|Service metric API (JSON format)|GET|/metrics-json|Obtains the dynamic average values of Time To First Token (TTFT) and Time Between Tokens (TBT) of nearly 1,000 requests by default, the number of requests being executed, the number of requests waiting, and the number of remaining NPU blocks during an inference service.|Native|
|Querying service management and control metrics (Prometheus format)|GET|/metrics|Queries management and control metrics of inference servitization.|Native|
|Dynamically loading LoRA|POST|/v1/load_lora_adapter|Dynamically loads LoRA.|OpenAI|
|Dynamically unloading LoRA|POST|/v1/unload_lora_adapter|Dynamically unloads LoRA.|OpenAI|

**Table 2** Model/Service query APIs (query APIs on the service plane)

|API|Interface Type|URL|Description|Framework|
|--|--|--|--|--|
|Model list|GET|/v1/models|Lists available models.|OpenAI|
|Model details|GET|/v1/models/{model}|Queries model information.|OpenAI|
|Service metadata query|GET|/v2|Obtains service metadata.|Triton|
|Model metadata query|GET|/v2/models/${MODEL_NAME}[/versions/\${MODEL_VERSION}]|Queries model metadata.|Triton|
|Model configuration query|GET|/v2/models/${MODEL_NAME}[/versions/\${MODEL_VERSION}]/config|Queries model configurations.|Triton|

**Table 3** Inference APIs (service APIs on the service plane)

|API|Interface Type|URL|Description|Framework|
|--|--|--|--|--|
|Inference Job|POST|/|TGI inference API. `stream==false` returns the text inference result, and `stream==true` returns the streaming inference result.|TGI|
|Inference Job|POST|/generate|Inference API of TGI and vLLM that uses request parameters to identify service types.|<ul><li>TGI</li><li>vLLM</li></ul>|
|Inference Job|POST|/generate_stream|TGI streaming inference API, which returns results in "Server-Sent Events" format.|TGI|
|Inference Job|POST|/v1/chat/completions|OpenAI text/streaming inference API.|OpenAI|
|Inference Job|POST|/v1/completions|vLLM-compatible OpenAI text/streaming inference API.|OpenAI|
|Inference Job|POST|/infer|Native inference API, which can return results in text or streaming mode.|Native|
|Inference Job|POST|/infer_token|Native inference API, which implements text or streaming inference based on input tokens.|Native|
|Inference Job|POST|/v2/models/${MODEL_NAME}[/versions/\${MODEL_VERSION}]/infer|Token inference API of Triton.|Triton|
|Inference Job|POST|/v2/models/${MODEL_NAME}[/versions/\${MODEL_VERSION}]/stopInfer|API for request termination in advance based on the Triton API definition.|Native|
|Inference Job|POST|/v2/models/${MODEL_NAME}[/versions/\${MODEL_VERSION}]/generate|Triton text inference API.|Triton|
|Inference Job|POST|/v2/models/${MODEL_NAME}[/versions/\${MODEL_VERSION}]/generate_stream|Triton streaming inference API.|Triton|
|Inference Job|POST|/v1/tokenizer|Calculation of the number of tokens.|Native|
|Inference Job|GET|/dresult|There is a persistent connection between the coordinator and the decode instance. Each time the decode instance generates an inference result, the result is returned to the coordinator through the persistent connection.|Prefill-decode disaggregation|

>[!NOTE]NOTE
>
>- The `${MODEL_NAME}` field specifies the name of the model to be queried.
>- The `[/versions/${MODEL_VERSION}]` field is not supported currently and is not passed.
