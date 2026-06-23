# Starting the haveged Service

## Prerequisites

The haveged service has been installed.

## Procedure

> [!NOTE]NOTE
>
> - The Server supports bidirectional HTTPS authentication to authenticate HTTP requests initiated by clients. When HTTPS is enabled, to encrypt the password of the server key, the random number generation algorithm used by the encryption tool KMC requires the haveged component to supplement entropy. Perform the following operations on all nodes where the Server is installed.
> - The MindIE MS server uses the KMC tool. The haveged component is also required for entropy supplementation on the management node where MindIE MS is deployed.
> - When MindIE Server is deployed by MindIE MS, the certificate is automatically generated in the container, which needs to be decrypted by KMC to generate a random password. This has high requirements on the entropy (4096). Therefore, the haveged component needs to be installed on the compute node to supplement the entropy.

Do as follows to check whether entropy needs to be added in the current Linux environment.

1. Check whether the haveged service is enabled in the system. (You are advised to keep the haveged service enabled.)

    ```bash
    systemctl status haveged.service
    ```

    Alternatively,

    ```bash
    ps -ef | grep "haveged" | grep -v "grep"
    ```

2. Change the entropy of the **/etc/default/haveged** configuration file to 4096.

    ```bash
    DAEMON_ARGS="-w 4096"
    ```

3. Start the haveged service and make it start with the system. Ensure that the haveged service is always started.

    ```bash
    systemctl start haveged.service
    systemctl enable haveged.service
    ```

4. Check the speed at which random numbers are displayed on the screen.

    ```bash
    cat /dev/random | od -x
    ```

    View the current entropy.

    ```bash
    cat /proc/sys/kernel/random/entropy_avail
    ```

    In normal cases, the entropy before haveged is started is over 100. After haveged is started, the entropy increases accordingly.
