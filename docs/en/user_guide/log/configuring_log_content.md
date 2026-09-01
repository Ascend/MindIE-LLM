# Configuring the Log Content

The environment variable  **MINDIE\_LOG\_VERBOSE**  is used to determine whether to print optional information in the log content of a component. The default value is  **true**, indicating that optional information is printed.

Format:  _Component name_: \{0, 1, true, false\}

- **0**  or  **false**  indicates no, and  **1**  or  **true**  indicates yes.
- If there is no component name before the colon \(:\), all components will be set in a unified manner by default.
- If multiple components are set at the same time, use semicolons \(;\) to separate them. The latest setting takes precedence and overwrites any previous configurations.

[**Example 1**] Do not print the optional log content of all  MindIE  components:

```bash
export MINDIE_LOG_VERBOSE="false"
```

[**Example 2**] Print the optional log content of  MindIE LLM:

```bash
export MINDIE_LOG_VERBOSE="llm: true"
```
