# Log Viewing

By default, MindIE collects logs of the Informational level or higher. The default log flush paths are described in [Table 1](#table1). For details about how to set the flush path, see [Setting the Log Flush Path](./setting_log_path.md).

**Table  1**  Log paths <a id="table1"></a>

| Path | Description |
|--|--|
| ~/mindie/log | Default log flush path. |
| ~/mindie/log/security | Security log path that is automatically generated in the default log flush path. |
| ~/mindie/log/debug | Run/Debug log path that is automatically generated in the default log flush path. |

The log file name format is mindie-_Component name_\_pid\_datetime.log. You can locate a log file based on the component name, process ID, and timestamp.

[**Example 1**] MindIE Motor CPP log file:

```text
mindie-service_123_202410080206.log
```

You can run the **cat** _Log file_ command to view logs.

[**Example 2**] View the MindIE Motor CPP log file:

```bash
cat mindie-service_123_202410080206.log
```
