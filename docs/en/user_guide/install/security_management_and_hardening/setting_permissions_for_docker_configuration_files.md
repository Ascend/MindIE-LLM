# Setting Permissions for Docker Configuration Files

- Setting permissions for the TLS CA certificate

    Set the owner and owner group of the TLS CA certificate file to `root:root`, and set the permission to `400`.

    The TLS CA certificate file (the path of the CA certificate file is specified by `--tlscacert`) needs to be protected from being tampered with. The certificate file is used to authenticate the Docker server. Therefore, the owner and owner group of the CA certificate must be `root`, and the permission must be 400 to ensure the integrity of the CA certificate.

    You can perform the following operations to set the file owner and permission:

    1. Run the following command to set the owner and owner group of the file to `root`:

        ```bash
        chown root:root {path to TLS CA certificate file}
        ```

        > [!NOTE]
        > Generally, `{path to TLS CA certificate file}` is `/usr/local/share/ca-certificates`.

    2. Set the file permission to `400`.

        ```bash
        chmod 400 {path to TLS CA certificate file}
        ```

- Setting permissions for `/etc/docker/daemon.json`

    Set the owner and owner group of the `daemon.json` file to `root:root`, and the file permission to `600`.

    The `daemon.json` file is an important global configuration file because it contains sensitive parameters for changing the Docker daemon. The owner and owner group of the file must be `root`, and only the `root` user has the write permission on the file to ensure file integrity. This file does not exist by default.

  - If the `daemon.json` file does not exist by default, the product does not use this file for configuration. In this case, you can run the following command to leave the configuration file empty in the boot parameters so that the file is not used as the default configuration file. This prevents attackers from maliciously creating and modifying configurations.

    ```bash
    docker --config-file=""
    ```

  - If the `daemon.json` file exists in the product environment, the file has been used for configuration. In this case, you need to set the corresponding permission to prevent malicious modification.
    1. Run the following command to set the owner and owner group of the file to `root`:

        ```bash
        chown root:root /etc/docker/daemon.json
        ```

    2. Run the following command to set the file permission to **600**:

        ```bash
        chmod 600 /etc/docker/daemon.json
        ```

        **Table 1** Permissions on Docker-related directories and files

        |Directory|File Owner|File Permission|
        |--|--|--|
        |/etc/default/docker|root:root|644 or higher|
        |/etc/sysconfig/docker|root:root|644 or higher|
        |docker.service|root:root|644|
        |docker.sock|root:docker|660|
        |/etc/docker|root:root|755 or higher|
        |docker.socket|root:root|644 or higher|

        > [!NOTE]
        > If the file or directory does not exist, ignore this part.
