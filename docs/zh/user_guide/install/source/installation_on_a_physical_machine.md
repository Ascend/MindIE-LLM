# 物理机安装方式

介绍通过物理机方式安装MindIE的操作步骤，本章以安装MindIE LLM为例。

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
   pip show mindie_llm | grep location
   ```

   若python版本是3.11，则默认安装路径为：`/usr/local/lib/python3.11/site-packages`
5. 执行如下命令，恢复权限。

   ```bash
   umask $old_umask
   ```
