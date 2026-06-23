# Containerized Installation

The following describes containerized installation. Before that, ensure that the server can connect to the network.

## Prerequisites

- You have installed Docker (version 24.x.x or later) on the host. For details about how to install Docker, see [Installing Docker](../source/docker_installation.md).
- Before configuring the source, make sure that the installation environment can connect to the network.

## Procedure

1. Pull the OS image.

   ```bash
   docker pull ubuntu:22.04
   ```

    Ubuntu 22.04 is used as an example. You may use other supported OS versions, but ensure the image complies with the hardware and OS requirements in [Hardware Requirements and Supported OSs](../installation_introduction.md#hardware-compatibility-and-supported-operating-systems).

    > [!NOTE]NOTE
    > - For torch_npu 2.1.0, pull Ubuntu 22.04; for torch_npu 2.9.0, pull Ubuntu 24.04 LTS.
    > - The APT source download path may be incorrect in a new container. You need to configure a dedicated source for Ubuntu 22.04 to improve the download speed.
    > - The installation requires the download of related dependencies. Ensure that the installation environment can be connected to the network.
    > - Run the `apt update` command as the·`root` user to check whether the source is valid.
    > - If an error is reported during command execution or dependency installation, check whether the network is connected, or replace the source in the `/etc/apt/sources.list` file with an available source or use an image source. (You can visit [Huawei open-source image website](https://mirrors.huaweicloud.com/) to find more information about how to configure a Huawei image source.)

2. Pull the container and mount host directories. During container installation, you do not need to install a driver in the container. You only need to mount the following directories to the container based on the product type.
 Start the container and modify the mounting information based on the actual product paths and requirements.

     ```bash
     docker run -it -d --net=host --shm-size=1g \
     --name <container-name> \
     --device=/dev/davinci_manager:rwm \
     --device=/dev/hisi_hdc:rwm \
     --device=/dev/devmm_svm:rwm \
     --device=/dev/davinci0:rwm \
     -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
     -v /usr/local/Ascend/firmware/:/usr/local/Ascend/firmware:ro \
     -v /usr/local/sbin:/usr/local/sbin:ro \
     -v /path-to-weights:/path-to-weights:ro \
     ubuntu:22.04 bash
     ```

     > [!NOTE]NOTE
     > For the `--device` parameter, the mount permission is set to `rwm` instead of the less permissive `rw` or `r`, for the following reasons:
        >- For the Atlas 800I A2 inference server, if the mount permission is set to `rw`, the container can be accessed normally, the `npu-smi` command can be used to view NPU usage, and MindIE services can run normally. However, if the mounted NPU (e.g., `davinci0` for `npu0` in mount options) is already occupied by another task, `npu-smi` will return an error and MindIE tasks will fail (e.g., `torch.npu.set_device()` will not work).
        >- For the Atlas 800I A3 SuperPoD server, if the mount permission is set to `rw`, entering the container and running `npu-smi` will print an error, MindIE tasks will fail, and `torch.npu.set_device()` will not work.

    **Table 1** Parameter description

     |Parameter|Description|
     |--|--|
     |--pids-limit -1|Removes the limit on the number of processes.<br>When using the Atlas 800I A2 inference server with Alibaba Cloud Linux 3.2104 U10, include this parameter in the container start command to disable the process limit.|
     |--shm-size=1g|Specifies the shared memory size (`/dev/shm`) for the container. Users can set a custom value—`1g` is an example.<br>The value cannot exceed the remaining physical memory of the host. You can run the `free -h` command to view the remaining physical memory. When data parallelism is enabled (DP > 1), the size of the shared memory needs to be adjusted as the DP value increases. <ul><li>For a DP value of 2, set `shm-size` to at least 2 GB;</li><li>For a DP value of 4, set `shm-size` to at least 3 GB.</li><li>For a DP value of 8, set `shm-size` to at least 5 GB.</li><li>For a DP value of 16, set `shm-size` to at least 9 GB.</li></ul>|
     |--name|Specifies the container name. Set it as required.|
     |--device|Mounts one or multiple devices.<br>The devices to be mounted are as follows: <ul><li>`/dev/davinci_manager`: Da Vinci-related management device. </li><li>`/dev/hisi_hdc`: HDC-related management device. </li><li>`/dev/devmm_svm`: Memory-related management device </li><li>`/dev/davinci0`: ID of the card to be mounted</li></ul><br>Run `ll /dev/ \| grep davinci` to check the number and names of devices. Then bind the required devices by modifying `--device=****`.|
     |-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro|Mounts the host directory `/usr/local/Ascend/driver` to the container. Change it according to the actual driver path.|
     |-v /usr/local/sbin:/usr/local/sbin:ro|Mounts the tools required in the container.|
     |-v /path-to-weights:/path-to-weights:ro|Mounts the directory where model weights on the host are located.|

3. Verify that the `npu-smi` tool is properly mounted (default path: `/usr/local/sbin/`; adjust as needed).
    1. Run the following command to list files in the directory and verify that the `npu-smi` tool exists:

        ```bash
        ll /usr/local/sbin/
        ```

    2. Check `npu-smi` permissions.

        Ensure the `npu-smi` binary has the proper execute permissions. Use the following command to update them:

        ```bash
        chmod 555 /usr/local/sbin/npu-smi
        ```

    3. Verify the execute permission.

        Run `npu-smi info` to check for output. If there is no output, double-check the previous steps.

        ```bash
        npu-smi info
        ```

4. Access the container.

    ```bash
    docker exec -it <container-name> /bin/bash
    ```

5. Configure `LD_LIBRARY_PATH` to include the `.so` files under `/usr/local/Ascend/driver/` as shown below:

    ```bash
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
    ```
