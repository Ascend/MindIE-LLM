# Setting the Log Display Mode

The environment variable **MINDIE\_LOG\_TO\_FILE** is used to specify whether to write logs of each  MindIE  component to files. The default value is **true**.

The environment variable **MINDIE\_LOG\_TO\_STDOUT** is used to specify whether to print logs of each  MindIE  component. The default value is **false**.

The format for specifying whether to write or print logs of a component:  _Component name_: \{0, 1, true, false\}

- **0** or **false** indicates no, and **1** or **true** indicates yes.
- If there is no component name before the colon \(:\), all components will be set in a unified manner by default.
- If multiple components are set at the same time, use semicolons \(;\) to separate them. The latest setting takes precedence and overwrites any previous configurations.

[**Example 1**] Do not write logs of  MindIE LLM  to files:

```bash
export MINDIE_LOG_TO_FILE="llm: false"
```

[**Example 2**] Print logs of all  MindIE  components:

```bash
export MINDIE_LOG_TO_STDOUT="true"
```
