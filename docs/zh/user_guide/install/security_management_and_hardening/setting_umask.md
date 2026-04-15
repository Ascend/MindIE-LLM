# 设置umask

建议用户将主机（包括宿主机）和容器中的umask设置为0027及以上，提高安全性。

以设置umask为0077为例，具体操作如下所示。

1. 以root用户登录服务器，编辑“/etc/profile“文件。

    ```bash
    vim /etc/profile
    ```

2. 在“/etc/profile“文件末尾加上**umask 0077**，保存并退出。
3. 执行如下命令使配置生效。

    ```bash
    source /etc/profile
    ```
