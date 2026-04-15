# 设置Docker配置文件权限

- **TLS CA证书权限配置**

    TLS CA证书文件属主和属组设为root:root，权限设为400。

    保护TLS CA证书文件（用参数--tlscacert指定CA证书文件的路径），防止其被篡改。证书文件用于指定的CA证书认证Docker服务器。因此，其属主和属组必须是root，权限必须为400，才能保证CA证书的完整性。

    可以通过以下方式设置。

    1. 执行以下命令，将文件的属主和属组设为root。

        ```bash
        chown root:root {path to TLS CA certificate file}
        ```

        > [!NOTE]说明
        > _\{path to TLS CA certificate file\}_路径一般为“/usr/local/share/ca-certificates”。

    2. 将文件权限设为400。

        ```bash
        chmod 400 {path to TLS CA certificate file}
        ```

- **“/etc/docker/daemon.json”文件权限配置**

    “daemon.json”文件属主和属组设为root:root，文件权限设为600。

    “daemon.json”文件包含更改Docker守护进程的敏感参数，是重要的全局配置文件，其属主和属组必须是root，且只对root可写，以保证文件的完整性，该文件并不是默认存在的。

  - 如果“daemon.json”文件默认不存在，说明产品没有使用该文件进行配置，那么可以执行以下命令，在启动参数中将配置文件设置为空，不使用该文件作为默认配置文件，避免被攻击者恶意创建并修改配置。

    ```bash
    docker --config-file=""
    ```

  - 如果产品环境存在“daemon.json”文件，说明已经使用了该文件进行配置操作，需要设置相应权限，防止被恶意修改。
    1. 执行以下命令，将文件的属主和属组设为root。

        ```bash
        chown root:root /etc/docker/daemon.json
        ```

    2. 执行以下命令，将文件权限设为600。

        ```bash
        chmod 600 /etc/docker/daemon.json
        ```

        **表 1**  Docker相关目录和文件权限控制

         |目录|文件属主|文件权限|
        |--|--|--|
        |/etc/default/docker|root:root|644或更严格|
        |/etc/sysconfig/docker|root:root|644或更严格|
        |docker.service|root:root|644|
        |docker.sock|root:docker|660|
        |/etc/docker|root:root|755或更严格|
        |docker.socket|root:root|644或更严格|

        > [!NOTE]说明
        > 文件或目录不存在时可忽略。
