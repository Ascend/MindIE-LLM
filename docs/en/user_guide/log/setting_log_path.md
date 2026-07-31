# Setting the Log Flush Path

The environment variable **MINDIE\_LOG\_PATH** is used to set the log flush path of each  MindIE  component. The default flush path is  **\~/mindie/log/debug**.

The format for setting the log flush path:_Component name_:_Path_

- If the path starts with a slash \(/\), the path is an absolute path.
- If the path does not start with a slash \(/\), the path is a relative path, which is relative to  **\~/mindie/log/debug**.
- If there is no component name before the colon \(:\), all components will be set in a unified manner by default.
- If multiple components are set at the same time, use semicolons \(;\) to separate them. The latest setting takes precedence and overwrites any previous configurations.

>[!NOTE]
>
>- The path cannot contain special characters.
>- The program does not check whether the log path contains soft links. Ensure that the log path is proper.

[**Example 1**] Flush  MindIE LLM  logs to  **/home/working/log/debug**:

```bash
export MINDIE_LOG_PATH="llm: /home/working/log/debug"
```

[**Example 2**] Flush  MindIE LLM  logs to  **\~/mindie/log/debug/llm**:

```bash
export MINDIE_LOG_PATH="llm: llm"
```
