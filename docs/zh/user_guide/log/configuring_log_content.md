# 配置日志内容

通过环境变量“MINDIE\_LOG\_VERBOSE”设置某个组件的日志内容中是否打印可选信息，默认为“true”打印可选信息。

设置的格式为：_组件名称_: \{0, 1, true, false\}。

- “0”和“false”代表否，“1”和“true”代表是。
- 如果“:”前无组件名称，则默认为对所有组件统一进行设置。
- 同时设置多个组件时用“;”隔开，且后方设置优先级高于前方设置，后方设置会覆盖前方设置。

【示例1】统一不打印或保存MindIE所有组件的可选日志内容。

```bash
export MINDIE_LOG_VERBOSE="false"
```

【示例2】打印或保存MindIE LLM的可选日志内容。

```bash
export MINDIE_LOG_VERBOSE="llm: true"
```
