# 配置兼容工具

## 功能概述

将老版本配置更新为指定的新版本配置。支持往前三个版本的配置兼容。

> [!NOTE]说明 
>-  不支持回退到低版本，不支持同版本更新配置，即target\_version必须大于源config.json的版本号。
>-  1.0.RC1中的“pipelineNumber“字段在后续版本中已删除，新增“modelInstanceNumber“字段代表模型实例个数，需要按照配置关系正确填写。

## 命令介绍

|参数|是否必选|说明|
|--|--|--|
|-h|-|显示帮助信息。需要单独使用，不能和别的参数一起使用。|
|--old_config_path|必选|老版本配置的绝对路径。|
|--old_version|必选|老版本的版本号。|
|--new_config_path|必选|转换到指定新版本的配置文件的绝对路径，默认生成文件为：*{脚本所在文件夹}*+*{系统时间戳}*+".json"。|
|--new_version|必选|新版本的版本号。|
|--upgrade_info_path|必选|新老版本配置文件之间的差异信息，文件路径为：*{脚本所在文件夹}下的*upgrade_info.json|
|--save_path|可选|老版本更新为新版本配置后，生成的新配置文件路径。如果未指定路径，默认路径为老版本的配置文件路径。|


## 操作步骤

1.  进入\{mindie-service\_install\_path\}/scripts/utils目录。

    ```bash
    cd {mindie-service_install_path}/scripts/utils
    ```

2.  转换版本配置。

    命令格式：

    ```bash
    python upgrade_server.py --old_config_path OLD_CONFIG_PATH --old_version OLD_VERSION --new_config_path NEW_CONFIG_PATH --new_version NEW_VERSION --upgrade_info_path UPGRADE_INFO_PATH [--save_path SAVE_PATH]
    ```

    以2.0.RC1版本配置更新为2.1.RC1版本配置为例，使用样例如下：

    ```python
    python upgrade_server.py --old_config_path ~/old/conf/config.json --old_version 2.0.RC1 --new_config_path ~/new/conf/config.json --new_version 2.1.RC1 --upgrade_info_path upgrade_info.json --save_path ~/new/conf/config.json
    ```

### 异常

如果实际需要转换的版本配置文件中的字段和对应版本包自带的默认配置文件中的字段名或格式不一致，那么转换会出现异常。

配置文件格式详情请参见各版本对应的用户手册。