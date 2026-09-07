# Combined Use and Activation Methods of Thinking, Enable_reasoning, Function Call, and Stream

This document describes how to enable and combine the four key features of MindIE: Thinking, Enable_reasoning, Function Call, and Stream. It covers their usage, priority, and interaction.

[Constraints] The Atlas 800I A2 inference server, Atlas 800I A3 SuperPoD server, and Atlas 300I duo inference card support the above features.

## Activation Method Overview

**Table 1** Summary of activation methods for the four features

| Dimension            | Activation Location                                                                               | Thinking                                                                              | Enable_reasoning                                            | Function Call                                                               | Stream                                     |
| -------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------ |
| **Request-level**    | Request body                                                                                      | Add `"chat_template_kwargs": {"enable_thinking": true/false}`                         | N/A                                                         | Pass `"tools": [...]` parameter. The model decides to trigger tool calling. | `"stream": true`/`false` (default `false`) |
| **Service-level**    | Service configuration file: `/usr/local/lib/python3.11/site-packages/mindie_llm/conf/config.json` | N/A                                                                                   | Configure `"enable_reasoning": true`/`false` under `models` | N/A                                                                         | N/A                                        |
| **Weight dimension** | `tokenizer_config.json` file in the model weight directory                                        | Add `"enable_thinking": true/false` (field names vary by model, see details below) | N/A                                                         | N/A                                                                         | N/A                                        |

---

## 1. Thinking

The Thinking feature controls whether the model outputs the thought process. It can be configured at the request-level and weight dimension.

### 1.1 Priority Description

**Priority: Request-level > Weight dimension > Model default behavior**

| Scenario                                                                          | Behavior Description                                                                                                                    |
| --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `enable_thinking` configured at request level                                     | Request-level configuration takes precedence.                                                                                           |
| `enable_thinking` configured at weight dimension, not configured at request level | Weight dimension configuration takes precedence.                                                                                        |
| Not configured at either level                                                    | Depend on model default behavior (e.g., Qwen3 enables Thinking by default, DeepSeek-V3.1 and DeepSeek-V3.2 disable Thinking by default) |

### 1.2 Request-Level Activation Method

Add the `chat_template_kwargs` field in the request body, and use the `enable_thinking` parameter to control enabling or disabling the Thinking feature.

**Request Example:**

```json
{
  "model": "your-model",
  "messages": [
    {
      "role": "user",
      "content": "Hello"
    }
  ],
  "chat_template_kwargs": {
    "enable_thinking": true
  }
}
```

**Configuration Description:**

| Field Configuration        | Description                                                                                                                              |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `"enable_thinking": true`  | Enables the Thinking feature.                                                                                                            |
| `"enable_thinking": false` | Disables the Thinking feature.                                                                                                           |
| Not adding this field      | Refer to the Weight Dimension configuration; if the Weight Dimension is also not configured, it depends on the model's default behavior. |

### 1.3 Weight Dimension Activation Method

Add the `enable_thinking` parameter in the `tokenizer_config.json` file under the model weight directory.

**Configuration Example:**

```json
{
  "enable_thinking": true
}
```

**Configuration Description:**

| Field Configuration        | Description                              |
| -------------------------- | ---------------------------------------- |
| `"enable_thinking": true`  | Enables the Thinking feature.            |
| `"enable_thinking": false` | Disables the Thinking feature.           |
| Not adding this field      | Depends on the model's default behavior. |

> **Note:** Different models have different configuration fields:
>
> - Qwen series: `"enable_thinking": true/false`
> - DeepSeek-V3.2: `"thinking": true/false`

## II. Enable_reasoning

The Enable_reasoning feature is used to separate the model's thinking process from the final answer, storing them in the `reasoning_content` and `content` fields, respectively.

### 2.1 Priority Description

Enable_reasoning is configured only at the **service-level**, with no conflicts from other dimensions.

### 2.2 Service-Level Activation Method

Add the `enable_reasoning` parameter under `ModelConfig` -> `models` in the serving configuration file `config.json`.

**Configuration Path:** `/usr/local/lib/python3.11/site-packages/mindie_llm/conf/config.json`

**Configuration Example:**

```json
"ModelDeployConfig": {
    "ModelConfig": [
        {
            "modelInstanceType": "Standard",
            "modelName": "Qwen3-32B",
            "modelWeightPath": "/data/weight/Qwen3-32B",
            "worldSize": 1,
            "backendType": "atb",
            "models": {
                "qwen3": {
                    "enable_reasoning": true
                }
            }
        }
    ]
}
```

**Configuration Description:**

| Configuration Item | Value             | Description                                                                                  |
| ------------------ | ----------------- | -------------------------------------------------------------------------------------------- |
| `enable_reasoning` | `true`            | Enables the function, parsing the output into two fields: `reasoning_content` and `content`. |
| `enable_reasoning` | `false` (default) | Disables the function, outputting only the `content` field.                                  |

> **Note:** The configuration fields vary for different models:

- Qwen3-30B-A3B model: The `"qwen3"` field should be changed to `"qwen3_moe"`.

- DeepSeek-R1 model: The `"qwen3"` field should be changed to `"deepseekv2"`, and the `model_type` field in the weight file should be changed to `"deepseek_v3"`.

- DeepSeek-V3.2 model: The `"qwen3"` field should be changed to `"deepseek_v32"`.

## III. Function Call

The Function Call feature allows the model to invoke external tools or APIs, expanding the model's application capabilities.

### 3.1 Priority Description

The triggering of Function Call is controlled at the **request-level**, while the tool parsing method is configured at the **service-level**.

| Dimension     | Function                                                                        |
| ------------- | ------------------------------------------------------------------------------- |
| Request-level | Determines whether to trigger Function Call (by passing the `tools` parameter). |
| Service-level | Configures the tool parsing method (`tool_call_parser`).                        |

### 3.2 Request-Level Activation Method

Add the `tools` field in the request body to pass a list of available tools. The model will decide whether to trigger a tool call based on the user input.

**Request Example:**

```json
{
  "model": "your-model",
  "messages": [
    {
      "role": "user",
      "content": "Check the shipping status for order #12345."
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_delivery_status",
        "description": "Get the shipping status of an order.",
        "parameters": {
          "type": "object",
          "properties": {
            "order_id": {
              "type": "string",
              "description": "Order ID"
            }
          },
          "required": ["order_id"]
        }
      }
    }
  ]
}
```

**Configuration Description:**

| Field   | Description                                          |
| ------- | ---------------------------------------------------- |
| `tools` | Tool list, containing available function definitions |

### 3.3 Service-Level Activation Method

Add the `tool_call_options` -> `tool_call_parser` parameter under `ModelConfig` -> `models` in the serving configuration file `config.json` to configure the service-level tool parsing method.

> Note: Function call behavior varies by model—no service‑level enabling is required for Qwen series, while DeepSeek‑V3.1 mandates it. For the `tool_call_parser` parameter description, refer to [function_call (Parameter Description)](function_call.md#parameter-description).

**Configuration Path:** `/usr/local/lib/python3.11/site-packages/mindie_llm/conf/config.json`

**Configuration Example:**

```json
"ModelDeployConfig": {
    "ModelConfig": [
        {
            "modelInstanceType": "Standard",
            "modelName": "dsv31",
            "modelWeightPath": "/data/weight/DeepSeek-V3.1",
            "worldSize": 16,
            "backendType": "atb",
            "models": {
                "deepseek_v3": {
                    "tool_call_options": {
                        "tool_call_parser": "deepseek_v31"
                    }
                }
            }
        }
    ]
}
```

## IV. Stream

The Stream feature controls whether the model output is returned in a streaming manner.

### 4.1 Priority Description

Stream is configured only at the **request-level**, with no conflicts across other dimensions.

### 4.2 Request-Level Activation Method

Add the `stream` field in the request body to control streaming output.

**Request Example:**

```json
{
  "model": "your-model",
  "messages": [
    {
      "role": "user",
      "content": "Hello"
    }
  ],
  "stream": true
}
```

**Configuration Description:**

| Field Configuration   | Description                                |
| --------------------- | ------------------------------------------ |
| `"stream": true`      | Enables Stream.                            |
| `"stream": false`     | Disables Stream.                           |
| Not adding this field | Default value is `false`, i.e., non-stream |

## V. Feature Combination Notes

### 5.1 Supported Feature Combinations

| Feature Combination                                  | Support Status         | Output Style Description                                                                                                                                                     |
| ---------------------------------------------------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Pairwise Combinations**                            |                        |                                                                                                                                                                              |
| Thinking + Thought Parsing                           | ✅ Supported            | Output separated into two fields:<br>• `reasoning_content`: thinking process<br>• `content`: final answer                                                                    |
| Thinking + Function Call                             | ✅ Supported            | Output includes:<br>• `content`: thinking process wrapped in `<think>...</think>` + description text.<br>• `tool_calls`: tool calling information (if a tool call is needed) |
| Thinking + Stream                                    | ✅ Supported            | Streaming output of `content`, outputting the thinking process `<think>...</think>` first, then the answer                                                                   |
| Enable_reasoning + Function Call                     | ✅ Supported            | Output includes:<br>• `content`: description text<br>• `tool_calls`: tool calling information                                                                                |
| Enable_reasoning + Stream                            | ✅ Supported            | Streaming output:<br>• `content`: answer                                                                                                                                     |
| Function Call + Stream                               | ⚠️ Partially Supported | Streaming output:<br>• `content`: description text<br>• `tool_calls`: tool calling information                                                                               |
| **Triple Combinations**                              |                        |                                                                                                                                                                              |
| Thinking + Enable_reasoning + Function Call          | ⚠️ Partially Supported | Output includes:<br>• `reasoning_content`: thinking process<br>• `content`: description text<br>• `tool_calls`: tool calling information                                     |
| Thinking + Enable_reasoning + Stream                 | ✅ Supported            | Streaming output:<br>• `reasoning_content`: thinking process<br>• `content`: answer                                                                                          |
| Thinking + Function Call + Stream                    | ⚠️ Partially Supported | Streaming output:<br>• `content`: thinking process + description text<br>• `tool_calls`: tool calling information                                                            |
| Enable_reasoning + Function Call + Stream            | ⚠️ Partially Supported | Streaming output:<br>• `content`: description text<br>• `tool_calls`: tool calling information                                                                               |
| **Quadruple Combination**                            |                        |                                                                                                                                                                              |
| Thinking + Enable_reasoning + Function Call + Stream | ⚠️ Partially Supported | Streaming output:<br>• `reasoning_content`: thinking process<br>• `content`: description text<br>• `tool_calls`: tool calling information                                    |

> Note: The content of the `<think>...</think>` tags included in `content` after the `Thinking` feature is enabled varies by model. For example, the `content` of Qwen3 series models contains `<think>...</think>` tags, while the `content` of the DeepSeek-V3.2 model contains `</think>` tags. Refer to the model deployment guide for specifics.

### 5.2 Constraints

- For `Enable_reasoning` constraints, refer to [enable_reasoning constraints](enable_reasoning.md#constraints).

- For `Function Call` constraints, refer to [function_call constraints](function_call.md#constraints).

- For models that support the above features, refer to the model deployment guide links in [Model List](../model_support_list.md).

## VI. Complete Configuration Example

### 6.1 Service-Level Configuration (config.json)

```json
{
    "ModelDeployConfig": {
        "ModelConfig": [
            {
                "modelInstanceType": "Standard",
                "modelName": "Qwen3-32B",
                "modelWeightPath": "/data/weight/Qwen3-32B",
                "worldSize": 1,
                "backendType": "atb",
                "trustRemoteCode": false,
                "models": {
                    "qwen3": {
                        "enable_reasoning": true
                    }
                }
            }
        ]
    }
}
```

### 6.2 Request Example (All Features)

```json
{
    "model": "your-model",
    "messages": [
        {
            "role": "user",
            "content": "Fetch today's weather forecast for Shanghai."
        }
    ],
    "chat_template_kwargs": {
        "enable_thinking": true
    },
    "tool_choice": "auto",
    "stream": true,
    "max_tokens": 1024,
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Fetch weather data for a specified city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City names, such as Beijing, Shenzhen, etc."
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]
}
```
