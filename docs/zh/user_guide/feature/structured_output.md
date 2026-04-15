# 结构化输出（Structured Output）

## 概述

结构化输出（Structured Output）是 MindIE LLM 提供的约束解码特性，允许用户指定模型输出必须严格符合特定格式（如 JSON Schema）。该特性基于xgrammar结构化约束后段，在推理阶段逐 token 约束模型的生成空间，确保输出结果为合法JSON格式可被直接解析使用。

**适用场景：**

- 需要模型返回可机器解析的 JSON 数据
- 需要输出字段名、类型、枚举值严格受控
- 下游系统对格式有强依赖（如工具调用结果解析、数据提取）

---

## 功能特性

| 特性 | 说明 |
|------|------|
| 约束后端 | xgrammar（基于 FSM 的高性能 token 约束库） |
| 支持格式类型 | `json_object`（通用 JSON 对象）、`json_schema`（用户指定 Schema） |

---

## 执行推理

以下介绍结构化输出在服务化场景下的使用方法。通过 OpenAI 兼容接口的 `response_format` 启用约束解码，支持 `POST http://{ip}:{port}/v1/chat/completions` 与 `POST http://{ip}:{port}/v1/completions`。

1. 启动服务。

    ```bash
    cd {MindIE安装目录}/latest/mindie-service/
    ./bin/mindieservice_daemon
    ```

    > [!NOTE]说明
    > 结构化输出在请求带有`response_format`参数时自动启用，无需在 config.json 中为本特性单独增加插件配置。

2. 向服务发送请求。参数说明见《MindIE Motor开发指南》中的「服务化接口 \> EndPoint业务面RESTful接口 \> 兼容OpenAI接口 \> 推理接口」章节。

    **json_object 模式**：要求模型输出任意合法的 JSON 对象 **（该模式下仅保证输出为合法 JSON；若需约束具体键与类型，请使用 json_schema 模式）**。

    **请求样例：**

    ```json
    curl -H "Content-type: application/json" -d '{
        "model": "dsv3_w8a8",
        "messages": [
            {
                "role": "user",
                "content": "提取以下文本中的关键信息并以 JSON 格式返回：张三，28岁，软件工程师，北京。"
            }
        ],
        "response_format": {
            "type": "json_object"
        },
        "stream": false,
        "max_tokens": 256
    }'  http://127.0.0.1:1025/v1/chat/completions
    ```

    **响应样例：**

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

    **json_schema 模式**：由用户指定 JSON Schema，模型输出须符合该 Schema。

    **请求样例：**

    ```json
    curl -H "Content-type: application/json" -d '{
        "model": "dsv3_w8a8",
        "messages": [
            {
                "role": "user",
                "content": "提取以下文本中的人员信息：李四，35岁，产品经理，上海，联系方式：13800138000。"
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
                            "description": "人员姓名"
                        },
                        "age": {
                            "type": "integer",
                            "description": "年龄"
                        },
                        "occupation": {
                            "type": "string",
                            "description": "职业"
                        },
                        "city": {
                            "type": "string",
                            "description": "城市"
                        },
                        "phone": {
                            "type": "string",
                            "description": "联系电话"
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

    **响应样例：**

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
                    "content":"{\"name\": \"李四\", \"age\": 35, \"occupation\": \"产品经理\", \"city\": \"上海\", \"phone\": \"13800138000\"}",
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

---

## 请求参数说明

`response_format` 参数结构如下：

### `json_object` 类型

| 字段 | 类型 | 是否必填 | 说明 |
|------|------|----------|------|
| `type` | string | 必填 | 固定值 `"json_object"` |

约束模型输出为任意合法的 JSON 对象。

### `json_schema` 类型

| 字段 | 类型 | 是否必填 | 说明 |
|------|------|----------|------|
| `type` | string | 必填 | 固定值 `"json_schema"` |
| `json_schema` | object | 必填 | Schema 描述对象 |
| `json_schema.name` | string | 必填 | Schema 名称（非空字符串，用于标识） |
| `json_schema.schema` | object | 可选 | 标准 JSON Schema 对象；不填时默认约束为通用 JSON 对象 |

`json_schema.schema` 遵循标准 JSON Schema 规范，支持以下关键字：

| 关键字 | 说明 |
|--------|------|
| `type` | 数据类型：`object`、`array`、`string`、`integer`、`number`、`boolean`、`null` |
| `properties` | 对象属性定义（`type: object` 时使用） |
| `required` | 必填属性列表 |
| `items` | 数组元素类型定义（`type: array` 时使用） |
| `enum` | 枚举值列表 |
| `description` | 属性描述（不影响约束，仅用于说明） |
| `additionalProperties` | 是否允许额外属性，默认 `false` |

---

## 限制与注意事项

1. 结构化输出特性支持在PD混部、PD分离场景下使用
2. 结构化输出特性支持与splitFuse、prefix_caching特性叠加使用
3. 结构化输出特性不支持与MTP与投机推理特性叠加使用
