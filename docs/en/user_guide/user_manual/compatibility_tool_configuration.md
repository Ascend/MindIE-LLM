# Configuration Compatibility Tool

## Function Overview

The tool can upgrade the configuration of an earlier version to that of a specified new version. The configurations of the previous three versions are compatible.

> [!NOTE]
>
>- Restoring the tool to an earlier version is not supported. The configuration cannot be updated within the same version. The `new_version` must be greater than the version number in the source `config.json`.

## Command Description

|Option|Required/Optional|Description|
|--|--|--|
|-h|-|Help information. This parameter must be used independently and cannot be used together with other parameters.|
|--old_config_path|Required|Absolute path configured in the earlier version.|
|--old_version|Required|Version number of the earlier version.|
|--new_config_path|Required|Absolute path of the configuration file converted to the specified new version. By default, the file is generated in the following format: *{Folder where the script is located}* + *{System timestamp}* + .json.|
|--new_version|Required|Version number of the new version.|
|--upgrade_info_path|Required|Differences between old and new configuration versions are recorded in `upgrade_info.json`, located in the script's directory.|
|--save_path|Optional|Path to the new configuration file generated after the configuration of the earlier version is updated to that of the new version. If the path is not specified, the default path is the configuration file path of the earlier version.|

## Procedure

1. Go to the `{mindie-service_install_path}/scripts/utils` directory.

    ```bash
    cd {mindie-service_install_path}/scripts/utils
    ```

2. Convert the version configuration.

    Command syntax:

    ```bash
    python upgrade_server.py --old_config_path OLD_CONFIG_PATH --old_version OLD_VERSION --new_config_path NEW_CONFIG_PATH --new_version NEW_VERSION --upgrade_info_path UPGRADE_INFO_PATH [--save_path SAVE_PATH]
    ```

    The following is an example of updating the configuration from version 2.0.RC1 to 2.1.RC1:

    ```bash
    python upgrade_server.py --old_config_path ~/old/conf/config.json --old_version 2.0.RC1 --new_config_path ~/new/conf/config.json --new_version 2.1.RC1 --upgrade_info_path upgrade_info.json --save_path ~/new/conf/config.json
    ```

### Exceptions

If the field names or formats in the version configuration file to be converted are different from those in the default configuration file of the corresponding version package, exceptions will occur during the conversion.

For details about the configuration file format, see the user guide of each version.
