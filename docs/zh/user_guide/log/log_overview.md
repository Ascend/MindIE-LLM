# 日志简介

## 日志分类

目前MindIE的日志按照内容分为安全审计日志和运行调试日志。任何与系统运行过程出现的登录认证、账户管理、访问控制、网络攻击等安全事件相关的信息都应该在安全审计日志中呈现。安全审计日志以外，在业务和调试过程中出现的日志为运行调试日志。

## 日志记录格式

MindIE所有组件的日志格式如下：

```text
[date time] [pid] [tid] [组件名称] [大写日志级别] [file:line] : [error code] [*] log message
```

> [!NOTE]说明
> \*：表示如果组件内有子组件或者更小的功能模块，会在日志信息前进行呈现。

**表 1**  日志字段说明

|字段|说明|
|--|--|
|**date time**|日期时间。|
|pid|进程号。|
|tid|线程号。|
|组件名称|MindIE的组件名称，有以下选项：[motor，server，llm，llmmodels，sd]。|
|**大写日志级别**|日志级别的大写形式，日志级别请参见[表1 日志级别](setting_log_level.md#table1)。|
|file:line|文件名:代码行号。|
|error code|Critical级别和部分Error级别日志的错误码，错误码请参见《MindIE错误码参考》。|
|**log message**|具体错误信息。|

**加粗内容为日志的必选内容**，其余字段为日志的可选信息，可以通过环境变量“MINDIE\_LOG\_VERBOSE”进行配置。具体操作请参见[配置日志内容](configuring_log_content.md)。
