# 物理机安装方式

指导用户进行MindIE物理机安装，请根据实际情况选择whl包或run包方式安装。

## whl包安装方式

介绍通过whl包方式安装MindIE的操作步骤，以下以安装MindIE LLM为例。

1. 为保证安装后的文件权限安全，请执行如下命令设置权限：

   ```bash
   old_umask=$(umask)
   umask 027
   ```

2. 执行如下命令，安装whl包。

   ```bash
   pip install mindie_llm-{version}-{python_tag}-{platform_tag}.whl --no-deps
   ```

   > [!NOTE]说明
   >
   > - 上方以mindie_llm包为例，如安装MindIE Motor或MindIE SD，请替换为对应的whl包名。
   > - 如果需要使用源码编译安装，请跳转到对应代码仓里获取编译指导。以MindIE-LLM为例，编译指导请[单击](https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/developer_guide/build_guide.md)。

3. 安装完成后，若打印如下信息，则说明软件安装成功：

    ```text
    Successfully installed xxx
    ```

   ```xxx``` 表示安装的实际软件包名。

4. (可选)执行如下命令，查询安装路径。

   ```bash
   pip show mindie_llm | grep Location
   ```

   若python版本是3.11，则默认安装路径为：`/usr/local/lib/python3.11/site-packages`
5. 执行如下命令，恢复权限。

   ```bash
   umask $old_umask
   ```

## run包安装方式

介绍通过run包方式安装MindIE的操作步骤，该方式将会依次安装MindIE Motor、MindIE LLM和MindIE SD各组件，组件包的路径在MindIE的子路径下。

1. 以CANN软件包的安装用户登录安装环境。
2. 将获取到的MindIE软件包上传到安装环境任意路径（如/home/package）。
3. 进入软件包所在路径。

    ```bash
    cd /home/package
    ```

4. 增加对软件包的可执行权限。

    ```bash
    chmod +x 软件包名.run
    ```

   **软件包名.run**表示开发套件包Ascend-mindie_\<version>\_linux-\<arch>_\<abi>.run，请根据实际包名进行替换。

5. 执行以下命令添加ascend-toolkit包的环境变量。（以root用户为例，以下为root用户的默认安装路径。）

    ```bash
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    ```

6. 执行以下命令校验软件包安装文件的一致性和完整性。

    ```bash
    ./软件包名.run --check
    ```

7. 执行以下命令安装软件（以下命令支持`-install-path=<path>`等参数，具体参数说明请参见[软件包参数说明](../faq_and_appendixes/software_package_options.md)）。

    ```bash
    ./软件包名.run --install --quiet
    ```

    > [!NOTE]说明
    >- 如果以root用户安装，**请勿安装在非root用户目录下**。
    >- 如果用户未指定安装路径，则软件会安装到默认路径下，默认安装路径如下。
    >   - root用户：“/usr/local/Ascend“
    >   - 非root用户：“/home/\{当前用户名\}/Ascend“
    >- 软件包安装详细日志路径如下。
    >   - root用户：“/var/log/mindie\_log/mindie\_install.log“
    >   - 非root用户：“/home/\{当前用户名\}/var/log/mindie\_log/mindie\_install.log“
    >- 安装过程中会在当前目录临时生成aie\_tmp\_source文件夹，安装完成后会删除，如果当前有同名文件夹会在安装后被删除。

    执行以上命令默认同意[华为企业业务最终用户许可协议（EULA）](https://e.huawei.com/cn/about/eula)的条款和条件。

    安装完成后，若打印如下信息，则说明软件安装成功：

    ```text
    xxx install success
    ```

    `xxx`表示安装的实际软件包名。

8. 配置环境变量
   当前提供进程级环境变量设置脚本，供用户在进程中引用，以自动完成环境变量设置。用户进程结束后自动失效。示例如下：

   root用户默认安装路径下配置环境变量：

   ```bash
   source /usr/local/Ascend/mindie/set_env.sh
   ```

   非root用户默认安装路径下配置环境变量：

   ```bash
   source /home/{当前用户名}/Ascend/mindie/set_env.sh
   ```

   用户也可以通过修改\~/.bashrc文件的方式设置永久环境变量，操作如下：

      a. 以运行用户在任意目录下执行`vi \~/.bashrc`命令，打开.bashrc文件，在文件最后一行后面添加上述内容。
      b. 执行`:wq!`命令保存文件并退出。
      c. 执行`source \~/.bashrc`命令使其立即生效。
