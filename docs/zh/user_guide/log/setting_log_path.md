# 配置日志落盘路径

通过环境变量“MINDIE\_LOG\_PATH”设置MindIE各组件日志的落盘路径，默认的落盘路径为“\~/mindie/log/debug”。

设置日志落盘路径的格式为：_组件名称_:  _路径_。

- 若路径开头为"/"，则表明该路径为绝对路径；
- 若路径开头无"/"，则表明该路径为相对路径，且是相对于“\~/mindie/log/debug”的路径。
- 如果“:”前无组件名称，则默认为对所有组件统一进行设置。
- 同时设置多个组件时用“;”隔开，且后方设置优先级高于前方设置，后方设置会覆盖前方设置。

> [!NOTE]说明
>
> - 路径里不能有特殊字符。
> - 程序不会校验日志路径是否包含软链接，请保证日志路径合理。

【示例1】将MindIE LLM的日志落盘到“/home/working/log/debug”。

```bash
export MINDIE_LOG_PATH="llm: /home/working/"
```

【示例2】将MindIE LLM的日志落盘到“\~/mindie/log/debug/llm”。

```bash
export MINDIE_LOG_PATH="llm: llm"
```
