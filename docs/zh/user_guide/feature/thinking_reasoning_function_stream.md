# Thinking、Enable_reasoning、Function Call、Stream 特性叠加及开启方式

本文档介绍 MindIE 中 Thinking（思考）、Enable_reasoning（思考解析）、Function Call（函数调用）和 Stream（流式输出）四大特性的开启方式、优先级关系及叠加使用说明。

【限制与约束】Atlas 800I A2 推理服务器、Atlas 800I A3 超节点服务器和 Atlas 300I Duo 推理卡支持。

## 特性开启方式总览

**表 1**  四大特性开启方式汇总

| 维度 | 开启位置 |是否开启思考 (Thinking) | 是否开启思考解析 (Enable_reasoning) | 是否开启 Function Call | 是否开启 Stream |
|------|------|------------------------|-----------------------------------|----------------------|----------------|
| **请求级** | 发送的请求体中 |添加`"chat_template_kwargs": {"enable_thinking": true/false}` | NA | 传入 `"tools": [...]` 参数，且模型决定触发工具调用 | `"stream": true/false`（默认 false） |
| **服务级** |服务化配置文件：`/usr/local/lib/python3.11/site-packages/mindie_llm/conf/config.json`  |NA |  `models` 下配置  `"enable_reasoning": true/false`  | NA | NA |
| **权重维度** | 模型权重目录下的 tokenizer_config.json文件 | 添加 `"enable_thinking": true/false`（不同模型字段名称不一样，详见下文） | NA | NA | NA | NA |

---

## 一、Thinking（思考）

Thinking 特性控制模型是否输出思考过程。支持在请求级和权重维度进行配置。

### 1.1 优先级说明

**优先级顺序：请求级 > 权重维度 > 模型默认行为**

| 场景 | 行为说明 |
|------|---------|
| 请求级配置了 `enable_thinking` | 以请求级配置为准 |
| 请求级未配置，权重维度配置了 `enable_thinking` | 以权重维度配置为准 |
| 均未配置 | 取决于模型默认行为（如 qwen3 默认开启思考，dsv3.1/dsv3.2 默认不开启思考） |

### 1.2 请求级开启方式

在请求体中添加 `chat_template_kwargs` 字段，通过 `enable_thinking` 参数控制思考功能的开启与关闭。

**请求示例：**

```json
{
  "model": "your-model",
  "messages": [
    {
      "role": "user",
      "content": "你好"
    }
  ],
  "chat_template_kwargs": {
    "enable_thinking": true
  }
}
```

**配置说明：**

| 字段配置 | 说明 |
|---------|------|
| `"enable_thinking": true` | 开启思考功能 |
| `"enable_thinking": false` | 关闭思考功能 |
| 不添加该字段 | 参考权重维度配置；如权重维度也未配置，则取决于模型默认行为 |

### 1.3 权重维度开启方式

在模型权重目录下的 `tokenizer_config.json` 文件中添加 `enable_thinking` 参数。

**配置示例：**

```json
{
  "enable_thinking": true
}
```

**配置说明：**

| 字段配置 | 说明 |
|---------|------|
| `"enable_thinking": true` | 开启思考功能 |
| `"enable_thinking": false` | 关闭思考功能 |
| 不添加该字段 | 是否思考取决于模型默认行为 |

> **注意：** 不同模型的配置字段不同：
> - qwen 系列：`"enable_thinking": true/false`
> - deepseekv3.2：`"thinking": true/false`

---

## 二、Enable_reasoning（思考解析）

Enable_reasoning 特性用于将模型的思考过程与最终回答进行分离，分别存储在 `reasoning_content` 和 `content` 字段中。

### 2.1 优先级说明

Enable_reasoning 仅在**服务级**进行配置，无其他维度冲突。

### 2.2 服务级开启方式

在服务化配置文件 `config.json` 的 `ModelConfig` -> `models` 下添加 `enable_reasoning` 参数。

**配置路径：** `/usr/local/lib/python3.11/site-packages/mindie_llm/conf/config.json`

**配置示例：**

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

**配置说明：**

| 配置项 | 取值 | 说明 |
|--------|---------|------|
| `enable_reasoning`  | `true`  | 开启模型思考解析，将输出分别解析为 `reasoning_content` 和 `content` 两个字段。默认值：`false` |
| `enable_reasoning`  | `false`（默认值） | 不开启模型思考解析，将输出`content` 字段。 |

> **注意：** 不同模型的配置字段不同：
- Qwen3-30B-A3B 模型：`"qwen3"` 字段应修改为 `"qwen3_moe"`
- DeepSeek-R1 模型：`"qwen3"` 字段应修改为 `"deepseekv2"`，并将权重文件中的 `model_type` 字段修改为 `"deepseek_v3"` 
- DeepSeek-V3.2 模型：`"qwen3"` 字段应修改为 `"deepseek_v32"`

---

## 三、Function Call（函数调用）

Function Call 特性允许模型调用外部工具或 API，扩展模型的应用能力。

### 3.1 优先级说明

Function Call 的触发由**请求级**控制，工具解析方式由**服务级**配置。

| 维度 | 作用 |
|------|------|
| 请求级 | 决定是否触发 Function Call（通过传入 `tools` 参数） |
| 服务级 | 配置工具解析方式（`tool_call_parser`） |

### 3.2 请求级开启方式

在请求体中添加 `tools` 字段，传入可用的工具列表。模型会根据用户输入决定是否触发工具调用。

**请求示例：**

```json
{
  "model": "your-model",
  "messages": [
    {
      "role": "user",
      "content": "查询订单 12345 的物流状态"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_delivery_status",
        "description": "获取订单的物流状态",
        "parameters": {
          "type": "object",
          "properties": {
            "order_id": {
              "type": "string",
              "description": "订单号"
            }
          },
          "required": ["order_id"]
        }
      }
    }
  ]
}
```

**配置说明：**

| 字段 | 说明 |
|------|------|
| `tools` | 工具列表，包含可用的函数定义 |

### 3.3 服务级开启方式


在服务化配置文件 `config.json` 的 `ModelConfig` -> `models` 下添加 `tool_call_options`-> `tool_call_parser` 参数，配置服务级工具解析方式。
>  注：不同的模型Function Call服务级开启方式不同，如Qwen系列模型无需服务级开启， DeepSeek-V3.1模型需要服务级开启，`tool_call_parser`参数说明请参见[function_call（参数说明）](function_call.md#参数说明)章节 。

**配置路径：** `/usr/local/lib/python3.11/site-packages/mindie_llm/conf/config.json`

**配置示例：**

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


---

## 四、Stream（流式输出）

Stream 特性控制模型输出是否采用流式方式返回。

### 4.1 优先级说明

Stream 仅在**请求级**进行配置，无其他维度冲突。

### 4.2 请求级开启方式

在请求体中添加 `stream` 字段控制流式输出。

**请求示例：**

```json
{
  "model": "your-model",
  "messages": [
    {
      "role": "user",
      "content": "你好"
    }
  ],
  "stream": true
}
```

**配置说明：**

| 字段配置 | 说明 |
|---------|------|
| `"stream": true` | 开启流式输出 |
| `"stream": false` | 关闭流式输出（非流式） |
| 不添加该字段 | 默认值为 `false`，即非流式输出 |

---

## 五、特性叠加说明

### 5.1 支持的特性叠加组合


| 特性组合 | 支持情况 | 输出样式说明 |
|---------|---------|-------------|
| **两两组合** |||
| Thinking + Enable_reasoning | ✅ 支持 | 输出分离为两个字段：<br>• `reasoning_content`: 思考过程<br>• `content`: 最终回答 |
| Thinking + Function Call | ✅ 支持 | 输出包含：<br>• `content`: 包含 `<think>...</think>` 包裹的思考过程 + 说明文字<br>• `tool_calls`: 工具调用信息（如需调用工具） |
| Thinking + Stream | ✅ 支持 | 流式输出 `content`，先输出思考过程 `<think>...</think>`，再输出回答 |
| Enable_reasoning + Function Call | ✅ 支持 | 输出包含：<br>• `content`: 说明文字<br>• `tool_calls`: 工具调用信息 |
| Enable_reasoning + Stream | ✅ 支持 | 流式输出：<br>• `content`: 回答内容 |
| Function Call + Stream | ⚠️ 部分支持 | 流式输出：<br>• `content`: 说明文字<br>• `tool_calls`: 工具调用信息 |
| **三三组合** |||
| Thinking + Enable_reasoning + Function Call | ⚠️ 部分支持 | 输出包含：<br>• `reasoning_content`: 思考过程<br>• `content`: 说明文字<br>• `tool_calls`: 工具调用信息 |
| Thinking + Enable_reasoning + Stream | ✅ 支持 | 流式输出：<br>• `reasoning_content`: 思考过程<br>• `content`: 回答内容 |
| Thinking + Function Call + Stream | ⚠️ 部分支持 | 流式输出：<br>• `content`: 包含思考过程 + 说明文字<br>• `tool_calls`: 工具调用信息 |
| Enable_reasoning + Function Call + Stream | ⚠️ 部分支持 | 流式输出：<br>• `content`: 说明文字<br>• `tool_calls`: 工具调用信息 |
| **四者组合** |||
| Thinking + Enable_reasoning + Function Call + Stream | ⚠️ 部分支持 | 流式输出：<br>• `reasoning_content`: 思考过程<br>• `content`: 说明文字<br>• `tool_calls`: 工具调用信息|

>  注：不同模型的`Thinking`特性开启后在`content`中包含的`<think>...</think>`标签内容不同，如Qwen3系列模型`content`中包含`<think>...</think>`标签， DeepSeek-V3.2模型`content`中包含`</think>`标签，具体请参考模型部署指导。

### 5.2 限制与约束
- `Enable_reasoning`限制与约束请参见[enable_reasoning 限制与约束](enable_reasoning.md#限制与约束)章节
- `Function Call`限制与约束请参见[function_call 限制与约束](function_call.md#限制与约束)章节
- 支持上述功能的模型详见[模型清单](../model_support_list.md)章节里各模型部署指导链接。
---

## 六、完整配置示例

### 6.1 服务级配置（config.json）

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

### 6.2 请求示例（全特性）

```json
{
    "model": "your-model",
    "messages": [
        {
            "role": "user",
            "content": "查询上海今天的天气"
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
                "description": "获取指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称，如北京、深圳等"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]
}
```