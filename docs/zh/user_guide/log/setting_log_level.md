# 设置日志级别

运行调试日志被分为如[表1](#table1)所示的5个等级。

**表 1**  日志级别 <a id="table1"></a>

|日志级别|简写|日志内容|
|--|--|--|
|CRITICAL|critical|紧急。系统业务严重受损或者完全不可用的紧急情况，规模性的用户受影响，需要运维人员紧急处理。例如系统无法启动或进程挂死等。|
|ERROR|error|错误。系统运行环境/功能受影响，或非预期的数据/事件造成功能执行出错。例如数据入库失败、任务创建失败等。|
|WARNING|warn|警告。系统出现的潜在风险或隐患，但不影响系统功能的正常执行。例如数据校验存在错误，但系统可通过纠错功能恢复，不影响功能的执行。|
|INFORMATIONAL|info|信息。用于系统运行正常的信息记录，输出一些状态或状态变化的信息，例如当前系统的状态、数据库的连接状态等信息。|
|DEBUG|debug|调试。用于跟踪运行路径，如跟踪函数的进入和退出等，记录调试信息。记载的信息全面，是给开发人员用于定位复杂的问题。增加了代码级的信息输出，如当前调用的函数名和参数、内部变量值、函数调用返回值等。抛出异常或者错误返回之前需要记录。|

日志级别等级由低到高顺序：DEBUG < INFORMATIONAL < WARNING < ERROR < CRITICAL，级别越低，输出日志越详细。

通过环境变量“MINDIE\_LOG\_LEVEL”设置各组件日志级别，日志级别默认为“info”。

设置某个组件日志级别的具体格式为：_组件名称_:  _日志级别_。

- 日志级别有以下选项：[critical, error, warn, info, debug]
- 组件名称有以下选项：[motor, server, llm, llmmodels, sd]
- 如果“:”前无组件名称，则默认为对所有组件统一进行设置。
- 同时设置多个组件日志级别时用“;”隔开，且后方设置优先级高于前方设置，后方设置会覆盖前方设置。

> [!NOTE]说明
> 以上组件和日志级别的取值不区分大小写。

【示例1】统一将MindIE所有组件的日志级别设成“debug”。

```bash
export MINDIE_LOG_LEVEL="debug"
```

【示例2】将MindIE所有组件的日志级别设成“debug”。

```bash
export MINDIE_LOG_LEVEL="llm:error ; debug"
```

【示例3】将MindIE LLM的日志级别设成“error”，将MindIE Client的日志级别设成“debug”。

```bash
export MINDIE_LOG_LEVEL="llm:error ; client:debug"
```

【示例4】除了Server的日志级别设成“debug”，其余组件的级别都设置为“info”。

```bash
export MINDIE_LOG_LEVEL="info ; server:debug"
```
