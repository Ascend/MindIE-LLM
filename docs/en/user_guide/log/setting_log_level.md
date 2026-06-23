# Setting the Log Level

The run/debug logs are classified into five levels, as described in [Table 1](#table1).

**Table  1**  Log levels <a id="table1"></a>

|Log Level|Short Form|Log Content|
|------|------|--------|
|CRITICAL|critical|Critical logs which record information when system services are severely damaged or completely unavailable and a large number of users are affected. In this case, O&M personnel need to handle the issue immediately. For example, the system cannot be started or the process is suspended.|
|ERROR|error|Error logs which indicate that the system operating environment or functions are affected, or function execution errors occur due to unexpected data or events. For example, data fails to be imported to the database or a task fails to be created.|
|WARNING|warn|Warning logs which indicate that the system has potential risks, but the risks do not affect system functions. For example, if an error occurs during data verification, the system rectifies this error using the error correction function, which does not affect the function execution.|
|INFORMATIONAL|info|Informational logs which record status and status change information when the system is running properly, for example, current system status and database connection status.|
|DEBUG|debug|Debug logs which record debugging information for message tracing such as function invocation tracing to help R&D personnel locate faults. Logs of this level also record code-level information such as the names of invoked functions, parameters, internal variable values, and return values of functions. Information must be recorded before an exception is thrown or an error is returned.|

The log levels are listed in ascending order as follows:**DEBUG**  <  **INFORMATIONAL**  <  **WARNING**  <  **ERROR**  <  **CRITICAL**. A lower level indicates more detailed output logs.

You can set the log level of each component using the environment variable  **MINDIE\_LOG\_LEVEL**. The default log level is  **info**.

The format for setting the log level of a component:_Component name_:  _Log level_

- The options of the log level are as follows: \[critical, error, warn, info, debug\]
- The options of the component name are as follows: \[motor, server, llm, sd\]
- If there is no component name before the colon \(:\), all components will be set in a unified manner by default.
- If multiple components are set at the same time, use semicolons \(;\) to separate them. The latest setting takes precedence and overwrites any previous configurations.

>[!NOTE]NOTE
>The components and log levels are case insensitive.

[**Example 1**] Set the log levels of all  MindIE  components to  **debug**  in a unified manner:

```bash
export MINDIE_LOG_LEVEL="debug"
```

[**Example 2**] Set the log levels of all  MindIE  components to  **debug**:

```bash
export MINDIE_LOG_LEVEL="llm:error ; debug"
```

[**Example 3**] Set the log level of  MindIE LLM  to  **error**  and the log level of  MindIE Client  to  **debug**:

```bash
export MINDIE_LOG_LEVEL="llm:error ; client:debug"
```

[**Example 4**] Set the log level of  Server  to  **debug**  and the log levels of other components to  **info**:

```bash
export MINDIE_LOG_LEVEL="info ; server:debug"
```
