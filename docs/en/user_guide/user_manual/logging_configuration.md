# MindIE Logging Guide

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-05-28T09:09:32.950Z pushedAt=2026-06-02T09:05:51.588Z -->

## Logging Configuration Change Notes

You can skip these configuration change notes and go directly to the [Logging Configuration Guide](#logging-configuration-guide)

### Version Compatibility

- **Change Date**: 2025-01-21

- **Scope of Impact**: All components using the MindIE logging system (LLM, LLMMODELS, SERVER)

### Unified Changes to Environment Variables

#### New Environment Variables

The following old environment variables have been removed. Use the new standardized variables instead.

| Old Environment Variable (To Be Deleted) | New Environment Variable (After Change) |
|:---------------------------------------- |:--------------------------------------- |
| **Log Printing**                         |                                         |
| `OCK_LOG_TO_STDOUT`                      | `MINDIE_LOG_TO_STDOUT`                  |
| `MINDIE_LLM_PYTHON_LOG_TO_STDOUT`        | `MINDIE_LOG_TO_STDOUT`                  |
| `MINDIE_LLM_LOG_TO_STDOUT`               | `MINDIE_LOG_TO_STDOUT`                  |
| `ATB_LOG_TO_STDOUT`                      | `MINDIE_LOG_TO_STDOUT`                  |
| `MIES_PYTHON_LOG_TO_STDOUT`              | `MINDIE_LOG_TO_STDOUT`                  |
| **Log Level Control**                    |                                         |
| `OCK_LOG_LEVEL`                          | `MINDIE_LOG_LEVEL`                      |
| `MINDIE_LLM_PYTHON_LOG_LEVEL`            | `MINDIE_LOG_LEVEL`                      |
| `MINDIE_LLM_LOG_LEVEL`                   | `MINDIE_LOG_LEVEL`                      |
| `ATB_LOG_LEVEL`                          | `MINDIE_LOG_LEVEL`                      |
| `LOG_LEVEL`                              | `MINDIE_LOG_LEVEL`                      |
| `MIES_PYTHON_LOG_LEVEL`                  | `MINDIE_LOG_LEVEL`                      |
| **Log Rotation Configuration**           |                                         |
| `MINDIE_LLM_PYTHON_LOG_MAXNUM`           | `MINDIE_LOG_ROTATE`                     |
| `MINDIE_LLM_PYTHON_LOG_MAXSIZE`          | `MINDIE_LOG_ROTATE`                     |
| **Log Write Path**                       |                                         |
| `MIES_PYTHON_LOG_PATH`                   | `MINDIE_LOG_PATH`                       |
| `MINDIE_LLM_PYTHON_LOG_PATH`             | `MINDIE_LOG_PATH`                       |
| **Log to File**                          |                                         |
| `MINDIE_LLM_PYTHON_LOG_TO_FILE`          | `MINDIE_LOG_TO_FILE`                    |
| `MINDIE_LLM_LOG_TO_FILE`                 | `MINDIE_LOG_TO_FILE`                    |
| `ATB_LOG_TO_FILE`                        | `MINDIE_LOG_TO_FILE`                    |
| `LOG_TO_FILE`                            | `MINDIE_LOG_TO_FILE`                    |
| `MIES_PYTHON_LOG_TO_FILE`                | `MINDIE_LOG_TO_FILE`                    |

#### `PYTHON_LOG_MAXSIZE` Compatibility Notes

**⚠️ Deprecation Notice**: `PYTHON_LOG_MAXSIZE` will be officially deprecated in **December 2026**. You can use `MINDIE_LOG_ROTATE` instead.

**🔄 Compatibility**: When both old and new environment variables are configured, `MINDIE_LOG_ROTATE` takes higher priority.

### Python-side Configuration Changes

| Configuration Item        | Before                                  | After                                   |
|:------------------------- |:--------------------------------------- |:--------------------------------------- |
| **Default Rotation Size** | 1GB                                    | 20MB (Synchronized with C++)           |
| **Number of Rotations**   | Fixed at 10                             | Configurable `[1, 64]`, default 10      |
| **Rotation File Suffix**  | `mindie-llm_{pid}_{datetime}.log.{num}` | `mindie-llm_{pid}_{datetime}.{num}.log` |

### Log File Naming and Path Changes

#### Python-side Component Log Consolidation Rules

**When `MINDIE_LOG_PATH` is set without distinguishing components**:

- The Python-side `llmmodels` component logs are no longer output separately to `mindie-llmmodels_{pid}_{datetime}.log`

- **Instead, they are merged into**  `mindie-llm_{pid}_{datetime}.log`

#### New Log File Types

##### C++

| Log File                                  | Content Description        |
|:----------------------------------------- |:-------------------------- |
| `mindie-llm-request_{pid}_{datetime}.log` | **Request processing log** |
| `mindie-llm-token_{pid}_{datetime}.log`   | **Token processing log**   |

##### Python

| Log File                                    | Content Description      |
|:------------------------------------------- |:------------------------ |
| `mindie-llm-token_{pid}_{datetime}.log`     | **Token processing log** |
| `mindie-llm-tokenizer_{pid}_{datetime}.log` | **Tokenizer log**        |

<a id="logging-configuration-guide"></a>

## Logging Configuration Guide

### Environment Variable Configuration

#### Basic Environment Variables

| Environment Variable | Function Description                                                         | Value Range                                                                                                                                                                                                                                                                                                   | Default Value             | Variable Status                     | Component-specific Configuration Support                                                                                                                  |
| -------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MINDIE_LOG_LEVEL     | Controls the log level.                                                      | DEBUG/INFO/WARN/ERROR/CRITICAL                                                                                                                                                                                                                                                                                | INFO                      | In Use                              | Yes                                                                                                                                                       |
| MINDIE_LOG_TO_FILE   | Controls whether logs are saved to a file.                                   | {0, 1, true, false}                                                                                                                                                                                                                                                                                           | true                      | In Use                              | Yes                                                                                                                                                       |
| MINDIE_LOG_TO_STDOUT | Controls whether logs are output to the terminal.                            | {0, 1, true, false}                                                                                                                                                                                                                                                                                           | false                     | In Use                              | Yes                                                                                                                                                       |
| MINDIE_LOG_PATH      | Controls the log write path.                                                 | N/A                                                                                                                                                                                                                                                                                                           | ~/mindie/log/debug        | In Use                              | Yes                                                                                                                                                       |
| MINDIE_LOG_VERBOSE   | Controls the log format.                                                     | {0, 1, true, false}                                                                                                                                                                                                                                                                                           | true                      | In Use                              | Yes                                                                                                                                                       |
| MINDIE_LOG_ROTATE    | Controls log rotation.                                                       | `-fs`: each process log file rotation size. When the log file exceeds this value, the current file is saved as an archive and a new file is created. Range: [1, 500].  <br/>`-r`: number of rotated log files to retain per process. Older files beyond this count are automatically deleted. Range: [1, 64]. | `-fs`: `20`<br>`-r`: `10` | In Use                              | Yes                                                                                                                                                       |
| PYTHON_LOG_MAXSIZE   | Controls the rotation size of each process log file  on the ATB Python side. | [0, 524288000] Bytes                                                                                                                                                                                                                                                                                          | None                      | To be discontinued in December 2026 | Only affects the ATB Python side, equivalent to "-fs" in "MINDIE_LOG_ROTATE". If both variables are configured, "MINDIE_LOG_ROTATE" takes higher priority |

Note:
`mindie-llm-token` logs are rotated at 1MB per process, with up to 2 retained files per process rotation. The `MINDIE_LOG_ROTATE` setting does not apply. These logs are written only to files, not to the terminal.

#### Log Format Description

##### Verbose Format (`MINDIE_LOG_VERBOSE=true`)

- Error level:

[Time] [Process ID] [Thread ID] [Component name] [Level] [File name-line number] : [Error code] Message content

- Other levels:

[Time] [Process ID] [Thread ID] [Component name] [Level] [File name-line number] : Message content

##### Simple Format (`MINDIE_LOG_VERBOSE=false`)

- Error level:

[Time] : [Level] [Error code] Message content

- Other levels:

[Time] : [Level] Message content

### Application Scenario Configuration

#### Serving Scenario

##### Log File Description

- Set `MINDIE_LOG_PATH` by component:
  
    (Example: `export MINDIE_LOG_PATH='llm:/path/to/llm_log;llmmodels:/path/to/llmmodels_log'`)
  
  | Component | File Name                                 | Content Description                |
  | --------- | ----------------------------------------- | ---------------------------------- |
  | llm       | mindie-llm_{pid}_{datetime}.log           | LLM service main log (cpp, python) |
  |           | mindie-llm-request_{pid}_{datetime}.log   | Request processing log (cpp)       |
  |           | mindie-llm-token_{pid}_{datetime}.log     | Token processing log (cpp, python) |
  |           | mindie-llm-tokenizer_{pid}_{datetime}.log | Tokenizer log (python)             |
  | llmmodels | mindie-llmmodels_{pid}_{datetime}.log     | Model management log (cpp, python) |
  | server    | mindie-server_{pid}_{datetime}.log        | Service management log (cpp)       |

- Set `MINDIE_LOG_PATH` without component distinction:
  
    (Example: `export MINDIE_LOG_PATH='/path/to/log'`)
    For the same Python process, logs from the `llmmodels` component (`mindie-llmmodels_{pid}_{datetime}.log`) and the `llm` component (`mindie-llm_{pid}_{datetime}.log`) will be written to the same file, consolidated as `mindie-llm_{pid}_{datetime}.log`.
  
  | Component       | File Name                                 | Content Description                                          |
  | --------------- | ----------------------------------------- | ------------------------------------------------------------ |
  | llm + llmmodels | mindie-llm_{pid}_{datetime}.log           | LLM service main log (python), Model management log (python) |
  | llm             | mindie-llm_{pid}_{datetime}.log           | LLM service main log (cpp)                                   |
  |                 | mindie-llm-request_{pid}_{datetime}.log   | Request processing log (cpp)                                 |
  |                 | mindie-llm-token_{pid}_{datetime}.log     | Token processing log (cpp, python)                           |
  |                 | mindie-llm-tokenizer_{pid}_{datetime}.log | Tokenizer log (python)                                       |
  | llmmodels       | mindie-llmmodels_{pid}_{datetime}.log     | Model management log (cpp)                                   |
  | server          | mindie-server_{pid}_{datetime}.log        | Service management log (cpp)                                 |

##### Configuration Example

**Recommended Configuration**

```bash
export MINDIE_LOG_LEVEL=INFO
export MINDIE_LOG_TO_FILE=1
export MINDIE_LOG_TO_STDOUT=0
```

**Basic Scenario Configuration**

```bash
# Set log level to INFO for all components.
export MINDIE_LOG_LEVEL=INFO

# Write logs to files for all components.
export MINDIE_LOG_TO_FILE=1

# Disable stdout logging for all components.
export MINDIE_LOG_TO_STDOUT=0

# Write all component logs to the specified directory.
export MINDIE_LOG_PATH='~/mindie/log/debug'

# Enable verbose log format for all components.
export MINDIE_LOG_VERBOSE=1

# Rotate logs: max 20MB per file, keep 10 rotated files.
export MINDIE_LOG_ROTATE='-fs 20 -r 10'
```

**Complex Scenario Configuration (Component-specific)**

```bash
# llm: debug; llmmodels: info
export MINDIE_LOG_LEVEL='llm:debug;llmmodels:info'

# llm: log to file; llmmodels: do not log to file
export MINDIE_LOG_TO_FILE='llm:true;llmmodels:false'

# llm: stdout logs; llmmodels: no stdout logs
export MINDIE_LOG_TO_STDOUT='llm:true;llmmodels:false'

# llm: log path '/path/to/llm_log'; llmmodels: log path '/path/to/llmmodels_log'
export MINDIE_LOG_PATH='llm:/path/to/llm_log;llmmodels:/path/to/llmmodels_log'

# llm: verbose log format; llmmodels: simple log format
export MINDIE_LOG_VERBOSE='llm:true;llmmodels:false'

# llm: max rotate size 1MB, 1 rotated file; llmmodels: max 2MB, 2 rotated files
export MINDIE_LOG_ROTATE='llm:-fs 1 -r 1;llmmodels:-fs 2 -r 2'

# For example, with the above configuration for the `llmmodels` component, log files are named as follows:
# mindie-llmmodels_{pid}_{datetime}.log       # active log file
# mindie-llmmodels_{pid}_{datetime}.01.log    # most recent rotated backup
# mindie-llmmodels_{pid}_{datetime}.02.log    # oldest rotated file (will be deleted)

# ATB Python (llmmodels): max rotated log size 4096 bytes
export PYTHON_LOG_MAXSIZE=4096
```

#### Pure Model Inference Scenario

##### Log File Description

| Component | File Name                             | Log Content     |
| --------- | ------------------------------------- | --------------- |
| llmmodels | mindie-llmmodels_{pid}_{datetime}.log | C++ side log    |
|           | mindie-llm_{pid}_{datetime}.log       | Python side log |

##### Configuration Example

**Recommended Configuration**

```bash
export MINDIE_LOG_LEVEL=INFO
export MINDIE_LOG_TO_FILE=1
export MINDIE_LOG_TO_STDOUT=1 # Enable this option for pure model inference to view the results.
```

### Special Notes

#### Supported Components

- **llm**: large language model service component

- **llmmodels**: model inference component

- **server**: service framework component

#### Performance Impact

- Enabling `MINDIE_LOG_TO_STDOUT` may impact inference performance.  

- In serving scenarios, it is recommended to keep it disabled and enable only when debugging.  

- For pure model inference, it is recommended to enable it only if you need to see the output; otherwise, keep it disabled.

#### Configuration Priority

- When both `PYTHON_LOG_MAXSIZE` and `MINDIE_LOG_ROTATE` are set, the `-fs` option of `MINDIE_LOG_ROTATE` takes precedence.

- If only `MINDIE_LOG_ROTATE` is set without `-fs`, the default value is `20`.

- If only `PYTHON_LOG_MAXSIZE` is set, it applies exclusively to ATB Python: each process log file rotation size uses `PYTHON_LOG_MAXSIZE`, while all other log rotation settings fall back to the defaults of `MINDIE_LOG_ROTATE`.

#### Log Rotation

- Log files are automatically rotated when they reach the specified size.

- Old files exceeding the backup count will be automatically deleted.
