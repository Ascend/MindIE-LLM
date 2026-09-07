# Structured Output

## Overview

Structured Output is a constrained decoding feature in MindIE LLM. It enforces model outputs to adhere to a strict format, such as JSON Schema. Built on the xgrammar constraint backend, this feature constrains the model's generation space token by token during inference, ensuring valid and directly parseable JSON output.

**Applicable Scenarios:**

- Require machine-parsable JSON output from the model.

- Require strictly controlled output field names, types, and enum values.

- Downstream systems have strong format dependencies (such as tool call result parsing and data extraction).

## Features

| Feature | Description |
|------|------|
| Constraint backend | xgrammar (a high-performance token constraint library based on FSM) |
| Supported format types | `json_object` (generic JSON object), `json_schema` (user-specified schema), and `text` (structured output not enabled; returned in natural language) |

## Inference

The following describes how to use Structured Output in serving scenarios. Constrained decoding is enabled via the `response_format` parameter of the OpenAI-compatible API, supporting `POST http://{ip}:{port}/v1/chat/completions` and `POST http://{ip}:{port}/v1/completions`.

1. Start the service.

    ```bash
    cd {MindIE_installation_directory}/latest/mindie-service/
    ./bin/mindieservice_daemon
    ```

    > [!NOTE]
    > Structured Output is automatically enabled when the request includes the `response_format` parameter. No additional plugin configuration is required in `config.json` for this feature.

2. Send a request to the service. For parameter descriptions, see  "Compatible with OpenAI APIs" \> "[Inference APIs](https://www.hiascend.com/document/detail/en/mindie/310/mindiellm/llmdev/mindie_llm0022.html)".

    **json_object mode**: Requires the model to output any valid JSON object **(This mode only guarantees the output is valid JSON; if you need to constrain specific keys and types, use the `json_schema` mode)**.

    **Request example:**

    ```json
    curl -H "Content-type: application/json" -d '{
        "model": "dsv3_w8a8",
        "messages": [
            {
                "role": "user",
                "content": "Extract key information from the following text and return as JSON: Zhang San, 28, software engineer, Beijing."
            }
        ],
        "response_format": {
            "type": "json_object"
        },
        "stream": false,
        "max_tokens": 256
    }'  http://127.0.0.1:1025/v1/chat/completions
    ```

    **Response example:**

    ```json
    {
        "id":"123456789",
        "object":"chat.completion",
        "created":1775112196,
        "model":"dsv3_w8a8",
        "choices":[
            {
                "index":0,
                "message":
                {
                    "role":"assistant",
                    "content":"{\"name\": \"Zhang San\", \"age\": 30, \"gender\": \"male\", \"occupation\": \"doctor\", \"workplace\": \"hospital\"}",
                    "tool_calls":[]
                },
                "logprobs":null,
                "finish_reason":"stop"
            }
        ],
        "usage":{
            "prompt_tokens":22,
            "prompt_tokens_details":{"cached_tokens":0},
            "completion_tokens":35,
            "completion_tokens_details":{"reasoning_tokens":0},
            "total_tokens":57
        }
    }
    ```

    **json_schema mode**: The user specifies a JSON Schema, and the model output must conform to that Schema.

    **Request example:**

    ```json
    curl -H "Content-type: application/json" -d '{
        "model": "dsv3_w8a8",
        "messages": [
            {
                "role": "user",
                "content": "Extract personnel info: Li Si, 35, Product Manager, Shanghai, Contact: 13800138000."
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "person_info",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name"
                        },
                        "age": {
                            "type": "integer",
                            "description": "Age"
                        },
                        "occupation": {
                            "type": "string",
                            "description": "Occupation"
                        },
                        "city": {
                            "type": "string",
                            "description": "City"
                        },
                        "phone": {
                            "type": "string",
                            "description": "Phone"
                        }
                    },
                    "required": ["name", "age", "occupation", "city", "phone"]
                }
            }
        },
        "stream": false,
        "max_tokens": 256
    }'  http://127.0.0.1:1025/v1/chat/completions
    ```

    **Response example:**

    ```json
    {
        "id": "12345678",
        "object": "chat.completion",
        "created": 1775112196,
        "model": "dsv3_w8a8",
        "choices":[
            {
                "index":0,
                "message":{
                    "role":"assistant",
                    "content":"{\"name\": \"Li Si\", \"age\": 35, \"occupation\": \"Product Manager\", \"city\": \"Shanghai\", \"phone\": \"13800138000\"}",
                    "tool_calls":[]
                },
                "logprobs":null,
                "finish_reason":"stop"
            }
        ],
        "usage":{
            "prompt_tokens":22,
            "prompt_tokens_details":{"cached_tokens":0},
            "completion_tokens":35,
            "completion_tokens_details":{"reasoning_tokens":0},
            "total_tokens":57
        }
    }
    ```

## Request Parameter Description

The `response_format` parameter structure is as follows:

### `json_object` Type

| Field | Type | Required/Optional | Description |
|------|------|----------|------|
| `type` | String | Required | Fixed value `"json_object"` |

Constrains the model to output any valid JSON object.

### `json_schema` Type

| Field | Type | Required/Optional | Description |
|------|------|----------|------|
| `type` | String | Required | Fixed value `"json_schema"` |
| `json_schema` | Object | Required | Schema description object |
| `json_schema.name` | String | Required | Schema name (non-empty string, used for identification) |
| `json_schema.schema` | Object | Optional | Standard JSON Schema object; defaults to a generic JSON object constraint if not specified |

`json_schema.schema` adheres to the standard JSON Schema specification and supports the following keywords:

| Keyword | Description |
|--------|------|
| `type` | Data type: `object`, `array`, `string`, `integer`, `number`, `boolean`, and `null` |
| `properties` | Object property definitions (used when `type: object`) |
| `required` | List of required properties |
| `items` | Array element type definition (used when `type: array`) |
| `enum` | List of enum values |
| `description` | Property description (does not affect constraints, for documentation only) |
| `additionalProperties` | Whether to allow additional properties, defaults to `false` |

## Constraints

1. The structured output feature supports use in PD co-location and PD disaggregation scenarios.

2. The structured output feature supports combined use with the SplitFuse and prefix cache features.

3. The structured output feature does not support combined use with the MTP and speculative inference features.
