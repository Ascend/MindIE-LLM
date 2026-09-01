# Setting umask

You are advised to set `umask` of the host (including the host machine) and container to `0027` or higher to improve security.

To set umask to `0077`, perform the following steps:

1. Log in to the server as the `root` user and edit the `/etc/profile` file.

    ```bash
    vim /etc/profile
    ```

2. Append `umask 0077` to the end of the file. Save the file and exit.
3. Make the configuration take effect.

    ```bash
    source /etc/profile
    ```
