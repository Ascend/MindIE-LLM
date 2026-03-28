# MindIE日志使用说明文档

## 日志配置变更说明

可跳过此配置变更说明，一键直达[日志配置指南](#日志配置指南)

### 版本兼容性

- **变更日期**: 2025-01-21
- **影响范围**: 所有使用 MindIE 日志系统的组件（LLM、LLMMODELS、SERVER）

### 环境变量统一变更

#### 新增环境变量

以下旧环境变量已被移除，请统一使用新的标准化变量：

| 旧环境变量（将被删除）              | 新环境变量（变更后）     |
| :-------------------------------- | :--------------------- |
| **日志是否打屏**                  |                        |
| `OCK_LOG_TO_STDOUT`               | `MINDIE_LOG_TO_STDOUT` |
| `MINDIE_LLM_PYTHON_LOG_TO_STDOUT` | `MINDIE_LOG_TO_STDOUT` |
| `MINDIE_LLM_LOG_TO_STDOUT`        | `MINDIE_LOG_TO_STDOUT` |
| `ATB_LOG_TO_STDOUT`               | `MINDIE_LOG_TO_STDOUT` |
| `MIES_PYTHON_LOG_TO_STDOUT`       | `MINDIE_LOG_TO_STDOUT` |
| **日志级别控制**                   |                        |
| `OCK_LOG_LEVEL`                   | `MINDIE_LOG_LEVEL`     |
| `MINDIE_LLM_PYTHON_LOG_LEVEL`     | `MINDIE_LOG_LEVEL`     |
| `MINDIE_LLM_LOG_LEVEL`            | `MINDIE_LOG_LEVEL`     |
| `ATB_LOG_LEVEL`                   | `MINDIE_LOG_LEVEL`     |
| `LOG_LEVEL`                       | `MINDIE_LOG_LEVEL`     |
| `MIES_PYTHON_LOG_LEVEL`           | `MINDIE_LOG_LEVEL`     |
| **日志轮转配置**                  |                        |
| `MINDIE_LLM_PYTHON_LOG_MAXNUM`    | `MINDIE_LOG_ROTATE`    |
| `MINDIE_LLM_PYTHON_LOG_MAXSIZE`   | `MINDIE_LOG_ROTATE`    |
| **日志写入路径**                  |                        |
| `MIES_PYTHON_LOG_PATH`            | `MINDIE_LOG_PATH`      |
| `MINDIE_LLM_PYTHON_LOG_PATH`      | `MINDIE_LOG_PATH`      |
| **日志是否写入文件**              |                        |
| `MINDIE_LLM_PYTHON_LOG_TO_FILE`   | `MINDIE_LOG_TO_FILE`   |
| `MINDIE_LLM_LOG_TO_FILE`          | `MINDIE_LOG_TO_FILE`   |
| `ATB_LOG_TO_FILE`                 | `MINDIE_LOG_TO_FILE`   |
| `LOG_TO_FILE`                     | `MINDIE_LOG_TO_FILE`   |
| `MIES_PYTHON_LOG_TO_FILE`         | `MINDIE_LOG_TO_FILE`   |

#### PYTHON_LOG_MAXSIZE 兼容性说明

**⚠️ 废弃通知**: `PYTHON_LOG_MAXSIZE` 将于 **2026年12月** 正式废弃，请改用 `MINDIE_LOG_ROTATE`。

**🔄 兼容性规则**: 新旧环境变量同时配置时`MINDIE_LOG_ROTATE` 优先级更高

### Python 侧配置变更

| 配置项 | 变更前 | 变更后   |
| :--- | :--- | :--- |
| **默认轮转大小** | 1GB | 20MB（同步 C++ 侧） |
| **轮转个数** | 固定 10 个 | 可配置 `[1, 64]`，默认 10个 |
| **轮转文件后缀** | `mindie-llm_{pid}_{datetime}.log.{num}` | `mindie-llm_{pid}_{datetime}.{num}.log` |

### 日志文件命名与路径变更

#### Python侧组件日志合并规则

**当不区分组件设置 `MINDIE_LOG_PATH` 时**：

- Python 侧 `llmmodels` 组件日志不再单独输出到 `mindie-llmmodels_{pid}_{datetime}.log`
- **统一合并**到 `mindie-llm_{pid}_{datetime}.log`

#### 新增日志文件类型

##### C++ 侧新增

| 日志文件 | 内容说明 |
| :--- | :--- |
| `mindie-llm-request_{pid}_{datetime}.log` | **请求处理日志** |
| `mindie-llm-token_{pid}_{datetime}.log` | **Token 处理日志** |

##### Python 侧新增

| 日志文件 | 内容说明 |
| :--- | :--- |
| `mindie-llm-token_{pid}_{datetime}.log` | **Token 处理日志** |
| `mindie-llm-tokenizer_{pid}_{datetime}.log` | **分词器日志** |

---

<a id="日志配置指南"></a>

## 日志配置指南

### 环境变量配置

#### 基础环境变量

| 环境变量名 | 功能描述 | 取值范围 | 默认值 | 变量状态 | 分组件配置支持 |
|-----------|---------|---------|--------|---------|--------------|
| MINDIE_LOG_LEVEL | 控制日志级别 | DEBUG/INFO/WARN/ERROR/CRITICAL | INFO | 正在使用 | 是 |
| MINDIE_LOG_TO_FILE | 控制日志是否保存到文件 | {0, 1, true, false} | true | 正在使用 | 是 |
| MINDIE_LOG_TO_STDOUT | 控制日志是否输出到终端 | {0, 1, true, false} | false | 正在使用 | 是 |
| MINDIE_LOG_PATH | 控制日志写入路径 | N/A | ~/mindie/log/debug | 正在使用 | 是 |
| MINDIE_LOG_VERBOSE | 控制日志格式 | {0, 1, true, false} | true | 正在使用 | 是 |
| MINDIE_LOG_ROTATE | 控制日志轮转 |-fs：每个进程日志文件轮转最大大小，当日志文件大小超过此数值时，当前文件会被保存为旧文件，并创建新文件继续记录日志，单位MB，取值范围[1, 500]<br>-r：每个进程轮转时可保留日志文件数量，超过此数量的旧日志文件会被自动删除，取值范围[1, 64] | -fs 20<br>-r 10 | 正在使用 | 是 |
| PYTHON_LOG_MAXSIZE | ATB Python每个进程日志文件轮转最大大小 | [0, 524288000] 字节 | None | 将于2026年12月日落 | 仅作用ATB Python侧，与“MINDIE_LOG_ROTATE”中的“-fs”等效，若两个变量同时配置，“MINDIE_LOG_ROTATE”优先级更高 |

注：  
mindie-llm-token默认每个进程日志文件轮转最大大小1MB，每个进程轮转时可保留日志文件数量2个，不受MINDIE_LOG_ROTATE控制；  
mindie-llm-token日志仅可写入文件，不会输出到终端；

#### 日志格式说明

##### 详细格式（MINDIE_LOG_VERBOSE=true）

- error级别：
[时间] [进程ID] [线程ID] [组件名] [级别] [文件名-行号] : [错误码] 消息内容
- 其他级别：
[时间] [进程ID] [线程ID] [组件名] [级别] [文件名-行号] : 消息内容

##### 简单格式（MINDIE_LOG_VERBOSE=false）

- error级别：
[时间] : [级别] [错误码] 消息内容
- 其他级别：
[时间] : [级别] 消息内容

### 应用场景配置

#### 服务化场景

##### 日志文件说明

- 分组件设置MINDIE_LOG_PATH：  
    （例如：export MINDIE_LOG_PATH='llm:/path/to/llm_log;llmmodels:/path/to/llmmodels_log'）

    | 组件 | 文件名 | 内容说明 |
    |------|--------|---------|
    | llm | mindie-llm_{pid}_{datetime}.log | LLM服务主日志(cpp、python) |
    | | mindie-llm-request_{pid}_{datetime}.log | 请求处理日志(cpp) |
    | | mindie-llm-token_{pid}_{datetime}.log | Token处理日志(cpp、python) |
    | | mindie-llm-tokenizer_{pid}_{datetime}.log | 分词器日志(python) |
    | llmmodels | mindie-llmmodels_{pid}_{datetime}.log | 模型管理日志(cpp、python) |
    | server | mindie-server_{pid}_{datetime}.log | 服务管理日志(cpp) |
 
- 不分组件设置MINDIE_LOG_PATH：  
    （例如：export MINDIE_LOG_PATH='/path/to/log'）  
    python侧同进程的llmmodels组件日志mindie-llmmodels_{pid}\_{datetime}.log会与llm组件日志mindie-llm_{pid}_
    {datetime}.log写在同一个日志文件中，归一为mindie-llm_{pid}_{datetime}.log

    | 组件 | 文件名 | 内容说明 |
    |------|--------|---------|
    | llm + llmmodels | mindie-llm_{pid}_{datetime}.log | LLM服务主日志(python)、模型管理日志(python) |
    | llm | mindie-llm_{pid}_{datetime}.log | LLM服务主日志(cpp) |
    | | mindie-llm-request_{pid}_{datetime}.log | 请求处理日志(cpp) |
    | | mindie-llm-token_{pid}_{datetime}.log | Token处理日志(cpp、python) |
    | | mindie-llm-tokenizer_{pid}_{datetime}.log | 分词器日志(python) |
    | llmmodels | mindie-llmmodels_{pid}_{datetime}.log | 模型管理日志(cpp) |
    | server | mindie-server_{pid}_{datetime}.log | 服务管理日志(cpp) |

##### 配置示例

**建议配置**

```bash
export MINDIE_LOG_LEVEL=INFO
export MINDIE_LOG_TO_FILE=1
export MINDIE_LOG_TO_STDOUT=0
```

**基础场景配置**

```bash
# 所有组件日志级别为INFO
export MINDIE_LOG_LEVEL=INFO

# 所有组件日志均写入文件
export MINDIE_LOG_TO_FILE=1

# 所有组件日志均不打屏
export MINDIE_LOG_TO_STDOUT=0

# 所有组件日志都写入指定目录
export MINDIE_LOG_PATH='~/mindie/log/debug'

# 所有组件日志均为详细格式
export MINDIE_LOG_VERBOSE=1

# 所有组件日志文件轮转大小最大 20 MB，轮转 10 个
export MINDIE_LOG_ROTATE='-fs 20 -r 10'

```

**复杂场景配置（分组件）**

```bash
# llm组件日志级别为debug；llmmodels组件日志级别为info
export MINDIE_LOG_LEVEL='llm:debug;llmmodels:info'

# llm组件日志写入文件；llmmodels组件日志不写入文件
export MINDIE_LOG_TO_FILE='llm:true;llmmodels:false'

# llm组件日志打屏；llmmodels组件日志不打屏
export MINDIE_LOG_TO_STDOUT='llm:true;llmmodels:false'

# llm组件日志写入路径'llm:/path/to/llm_log'；llmmodels组件日志写入路径'llm:/path/to/llmmodels_log'
export MINDIE_LOG_PATH='llm:/path/to/llm_log;llmmodels:/path/to/llmmodels_log'

# llm组件日志设置详细格式；llmmodels组件日志设置简单格式
export MINDIE_LOG_VERBOSE='llm:true;llmmodels:false'

# llm组件同进程日志文件轮转大小最大 1 MB，轮转 1 个；llmmodels组件同进程日志文件轮转大小最大 2 MB，轮转 2 个
export MINDIE_LOG_ROTATE='llm:-fs 1 -r 1;llmmodels:-fs 2 -r 2'
"""
例：
对于上述配置中的llmmodels组件，文件命名规则如下：
mindie-llmmodels_{pid}_{datetime}.log       # 当前正在写入的日志文件（保留）
mindie-llmmodels_{pid}_{datetime}.01.log    # 最新轮转的备份文件（保留）
mindie-llmmodels_{pid}_{datetime}.02.log    # 最早的轮转文件（会被删除））

"""

# ATB Python（llmmodels）同进程日志文件轮转大小最大 4096 Byte
export PYTHON_LOG_MAXSIZE=4096

```

#### 纯模型推理场景

##### 日志文件说明

| 组件 | 文件名 | 日志内容 |
|------|--------|---------|
| llmmodels | mindie-llmmodels_{pid}_{datetime}.log | cpp侧日志 |
| | mindie-llm_{pid}_{datetime}.log | python侧日志 |

##### 配置示例

**建议配置**

```bash
export MINDIE_LOG_LEVEL=INFO
export MINDIE_LOG_TO_FILE=1
export MINDIE_LOG_TO_STDOUT=1 # 纯模型推理时建议打开，可看到推理结果
```

### 特别说明

#### 支持的组件

- **llm**: 大语言模型服务组件
- **llmmodels**: 模型推理组件
- **server**: 服务化框架组件

#### 性能影响

- 开启MINDIE_LOG_TO_STDOUT会影响推理性能
- 服务化场景建议默认关闭，仅在需要调试时开启
- 纯模型推理场景看输出建议开启，否则关闭即可

#### 配置优先级

- 当“PYTHON_LOG_MAXSIZE”和“MINDIE_LOG_ROTATE”同时设置时，MINDIE_LOG_ROTATE的-fs参数优先级高于PYTHON_LOG_MAXSIZE
- 若只设置MINDIE_LOG_ROTATE且未配置-fs，使用默认值20MB
- 若只设置PYTHON_LOG_MAXSIZE，除ATB Python每个进程日志文件轮转最大大小为PYTHON_LOG_MAXSIZE外，其他日志文件轮转配置使用MINDIE_LOG_ROTATE默认值

#### 文件轮转

- 日志文件达到指定大小会自动轮转
- 超出备份数量的旧文件会被自动删除
