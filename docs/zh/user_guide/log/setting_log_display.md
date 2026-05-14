# 设置日志展示方式

通过环境变量“MINDIE\_LOG\_TO\_FILE”设置MindIE各组件日志是否写入文件，默认为“true”写入。

通过环境变量“MINDIE\_LOG\_TO\_STDOUT”设置MindIE各组件日志是否打印，默认为“false”不打印。

设置某个组件日志是否写入或打印的格式为：_组件名称_: \{0, 1, true, false\}。

- “0”和“false”代表否，“1”和“true”代表是。
- 如果“:”前无组件名称，则默认为对所有组件统一进行设置。
- 同时设置多个组件时用“;”隔开，且后方设置优先级高于前方设置，后方设置会覆盖前方设置。

【示例1】不将MindIE LLM的日志写入文件。

```bash
export MINDIE_LOG_TO_FILE="llm: false"
```

【示例2】将MindIE所有组件的日志流打印。

```bash
export MINDIE_LOG_TO_STDOUT="true"
```
