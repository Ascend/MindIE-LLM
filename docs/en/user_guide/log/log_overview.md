# Log Overview

## Log Type

Currently,  MindIE  logs are classified into security audit logs and run/debug logs. Any information that is related to login authentication, account management, access control, network attacks, and other security events during system running must be recorded in security audit logs. Except security audit logs, logs generated during service and debugging are run/debug logs.

## Log Format

Log format of all  MindIE  components:

```text
[date time] [pid] [tid] [Component name] [Log level in uppercase] [file:line] : [error code] [*] log message
```

>[!NOTE]
>The asterisk * indicates that if a component contains subcomponents or smaller functional modules, they are displayed prior to the log information.

**Table  1**  Log field description

| Field | Description |
|-------|-------------|
| **date time** | Date and time. |
| pid | Process ID. |
| tid | Thread ID. |
| Component name | Name of the MindIE component. Options: `[motor, server, llm, llmmodels, sd]`. |
| **Log level (uppercase)** | Uppercase log level. See [Table 1: Log Levels](setting_log_level.md#table1). |
| file:line | Filename and line number. |
| error code | Error code for Critical-level and some Error-level logs. See [MindIE Error Code Reference](https://www.hiascend.com/document/detail/en/mindie/310/ref/errorcodereference/mindie_log_0072.html). |
| **log message** | Detailed error message. |

**Bold fields are required** in the log; all other fields are optional and can be configured via the `MINDIE_LOG_VERBOSE` environment variable. For details, see [Configuring the Log Content](configuring_log_content.md).
