# 升级

## whl包安装方式

用户可通过安装新版本的whl包，完成MindIE升级。本章以升级MindIE LLM为例。

执行如下命令，安装新whl包，完成升级。

```bash
pip install mindie_llm-{version}-{python_tag}-{platform_tag}.whl

```

> [!NOTE]说明
> 上方以mindie_llm包为例，如升级MindIE Motor或MindIE SD，请替换为对应的whl包名。

如果是同版本升级，在如上命令中加 `--force-reinstall` 参数进行强制重新安装。

> [!CAUTION]注意
> 重装mindie_llm过程中，安装目录（/mindie_llm）下的内容会被全部删除再重装新版本，如果配置文件、证书文件等需要保留，请先备份。

## run包安装方式

MindIE包的升级请在获取升级版本的软件包后，执行`--upgrade`命令覆盖旧版本。

1. 以软件包的安装用户登录软件包的安装环境。
2. 进入软件包所在路径。
3. 增加对软件包的可执行权限。

    ```bash
    chmod +x 软件包名.run
    ```

4. 执行如下命令校验软件包安装文件的一致性和完整性。

    ```bash
    ./软件包名.run --check
    ```

5. 软件包升级。

    - 若用户安装时指定了安装路径，执行命令如下：

        ```bash
        ./软件包名.run --upgrade --install-path=<path>
        ```

        其中\<path>为用户指定的软件包安装目录，软件包名请根据实际包名进行替换。

    - 若用户安装时未指定安装路径，执行命令如下：

        ```bash
        ./软件包名.run --upgrade
        ```

    > [!NOTE]说明
        > 如果用户未指定升级路径，则软件会升级到默认路径下，默认升级路径如下。
        >-  root用户："/usr/local/Ascend"
        >-  非root用户："/home/\{当前用户名\}/Ascend"

    用户想使用默认签署[华为企业业务最终用户许可协议（EULA）](https://e.huawei.com/cn/about/eula)的方式升级软件包时，可以添加`--quiet`参数配合升级命令使用，如：`./软件包名.run --upgrade--quiet`添加该参数后会跳过[6](#step6)的确认操作。

6. <a id="step6"></a>用户需签署[华为企业业务最终用户许可协议（EULA）](https://e.huawei.com/cn/about/eula)后进入升级流程，根据回显页面执行**y或Y**确认协议，输入其他任意字符为拒绝协议，确认接受协议后开始升级。

    若当前语言环境不满足要求，可以执行如下命令配置系统的默认语言环境。

    ```bash
    #配置为中文（简体）
    export LANG=zh_CN.UTF-8
    #配置为英文
    export LANG=en_US.UTF-8
    ```

7. 软件包升级完成后，请参考以下方式更新依赖，两种方式任选其一。

    - 方式一：参考[安装依赖](installing_software_and_dependencies.md#安装依赖)章节安装缺失依赖以及版本存在变更的依赖。

        例如：MindIE 2.0.RC1相对于1.0.0版本，新增了numba依赖，则使用以下命令安装该依赖。

        ```bash
        pip3 install numba==0.61.2
        ```

    - 方式二：参考[安装依赖](./installing_software_and_dependencies.md#安装依赖)章节，使用以下命令安装当前版本所需的全量依赖。（该方式可能会将环境中已制定的依赖版本覆盖）

        ```bash
        pip3 install -r requirements.txt
        ```
