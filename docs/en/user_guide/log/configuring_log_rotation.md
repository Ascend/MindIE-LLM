# Configuring Log Rotation

When the file size exceeds the product of the size of each log file and the number of log files that can be written by each process, log rotation starts. The environment variable  **MINDIE\_LOG\_ROTATE**  is used to rotate logs.

The size of each log file ranges from 1 MB to 500 MB, and the default value is 20 MB. The number of log files that can be written by each process ranges from 1 to 64, and the default value is 10.

The format for rotating logs of a component:  _Component name_: -fs  _filesize_  -r  _rotate_

- If there is no component name before the colon \(:\), all components will be set in a unified manner by default.
- If multiple components are set at the same time, use semicolons \(;\) to separate them. The latest setting takes precedence and overwrites any previous configurations.
- _filesize_  indicates the size of each log file, and  _rotate_  indicates the number of log files that can be written by each process.

    >[!NOTE]NOTE 
    >The naming format for log rotation and aging: mindie-_Component name_\__Process ID_\__Time_._Rotation file ID_.log

[Example 1] Set the size of each log file of all  MindIE  components to 500 MB:

```bash
export MINDIE_LOG_ROTATE="-fs 500"
```

[Example 2] Set the maximum size of the  MindIE Torch  log file to 40 MB and allow each process to write one log file:

```bash
export MINDIE_LOG_ROTATE="torch: -fs 40 -r 1"
```
