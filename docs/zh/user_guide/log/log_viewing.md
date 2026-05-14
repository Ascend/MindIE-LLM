# 查看日志

MindIE默认收集Informational级别及以上的日志，日志文件的默认落盘路径如[表1](#table1)所示。落盘路径的设置可参见[配置日志落盘路径](setting_log_path.md)。

**表 1**  日志路径 <a id="table1"></a>

|路径|说明|
|--|--|
|~/mindie/log|默认的日志落盘路径。|
|~/mindie/log/security|默认日志落盘路径下，自动生成的安全日志路径。|
|~/mindie/log/debug|默认日志落盘路径下，自动生成的运行调试日志路径。|

日志文件命名格式统一为：mindie-_组件名称_\_pid\_datetime.log。可以根据组件名称，进程号，和时间戳来定位到相关的日志文件。

【示例1】MindIE Motor的日志文件。

```text
mindie-service_123_202410080206.log
```

使用如下命令，查看日志：cat  _日志文件_

【示例2】查看MindIE Motor的日志文件。

```bash
cat mindie-service_123_202410080206.log
```
