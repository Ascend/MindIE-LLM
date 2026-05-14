# 卸载

## whl包安装方式

用户可通过卸载whl软件包完成卸载，本章以卸载MindIE LLM为例。

执行如下命令，卸载whl包，完成卸载。

```bash
pip uninstall mindie_llm-{version}-{python_tag}-{platform_tag}.whl

```

> [!NOTE]说明
> 上述以mindie_llm包为例，如卸载MindIE Motor或MindIE SD，请替换为对应的whl包名。

卸载完成后，若输出如下信息，则说明软件卸载成功：

```bash

[INFO] XXX uninstall success

```

## run包安装方式

用户可以通过运行卸载脚本和卸载软件包完成卸载。

### 方法一：脚本式卸载（推荐方式）

用户可以通过卸载脚本完成卸载。

1. 进入软件的卸载脚本所在路径，一般放置在“scripts“目录下（“scripts”所在路径请以实际为准）。

    ```bash
    cd <path>/mindie/<version>/scripts
    ```

    其中<path\>为软件包的安装路径，<version\>为软件包版本，请用户根据实际情况替换。

2. 执行`./uninstall.sh`命令运行脚本，完成卸载。

    卸载完成后，若打印如下信息，则说明软件卸载成功：

    ```text
    [INFO] xxx uninstall success
    ```

    `xxx`表示卸载的实际软件包名。

    > [!NOTE]说明
    > 卸载完成后，建议使用以下命令取消TUNE\_BANK\_PATH环境变量配置：
    >
    >```bash
    >unset TUNE_BANK_PATH
    >```

### 方法二：软件包卸载

如果用户想要对已安装的软件包进行卸载，可以执行如下步骤：

1. 以软件包的安装用户登录软件包的安装环境。
2. 进入软件包所在路径。
3. 执行以下命令卸载软件包。

    - 若用户安装时指定了安装路径，执行命令如下：

        ```bash
        ./软件包名.run --uninstall --install-path=<path>
        ```

    - 若用户安装时未指定安装路径，执行命令如下：

        ```bash
        ./软件包名.run --uninstall
        ```

    卸载完成后，若打印如下信息，则说明软件卸载成功：

    ```text
    [INFO] xxx uninstall success
    ```

    `xxx`表示卸载的实际软件包名。

    > [!NOTE]说明
    > 卸载完成后，建议使用以下命令取消TUNE\_BANK\_PATH环境变量配置：
    >
    >```bash
    >unset TUNE_BANK_PATH
    >```
