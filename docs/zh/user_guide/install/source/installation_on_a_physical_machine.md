# 物理机安装方式

介绍通过物理机方式安装MindIE的操作步骤，本章以安装MindIE LLM为例。

1. 为保证安装后的文件权限安全，请执行如下命令设置权限：
   
   ```
   old_umask=$(umask)
   umask 027
   ```

2. 执行如下命令，安装whl包。
   
   ```
   pip install mindie_llm-{version}-{python tag}-{abi tag}-{platform tag}.whl --no-deps
   ```
   > [!NOTE]说明
   > 上方以mindie_llm包为例，如安装MindIE Motor或MindIE SD，请替换为对应的whl包名。

3. 执行如下命令，恢复权限。
   
   ```
   umask $old_umask
   ```
