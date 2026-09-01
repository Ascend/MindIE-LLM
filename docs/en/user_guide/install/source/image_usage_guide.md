# Image Deployment Mode

The following describes how to install the MindIE container image. Before that, ensure that the server can connect to the network.

## Prerequisites

- Ensure that the NPU driver and firmware have been installed on the host. If the firmware and driver are not installed, download [firmware and driver](https://www.hiascend.com/hardware/firmware-drivers) and select the firmware and driver of the community edition or commercial edition based on the product series and model. Run the following commands to install them:
  
    ```shell
    chmod +x Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run
    chmod +x Ascend-hdk-<chip_type>-npu-firmware_<version>.run
    ./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --full --force
    ./Ascend-hdk-<chip_type>-npu-firmware_<version>.run --full
    ```

- You have installed Docker (version 24.x.x or later) on the host. For details about how to install Docker, see [Installing Docker](https://gitcode.com/Ascend/MindIE-LLM/blob/v3.1.0/docs/zh/user_guide/install/source/docker_installation.md).
- Before configuring the source, make sure that the installation environment can connect to the network.

> [!NOTE]
> For the Atlas 200I Pro acceleration module, the host OS and container image OS must be compatible as follows:
>
> - Ubuntu 22.04 hosts support running Ubuntu 24.04 container images.
> - openEuler 22.03 hosts support running openEuler 24.03 container images.
>
> Choose a container image version that is compatible with your host operating system.

## Obtaining the MindIE Image

1. Click [AscendHub](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f) to go to the MindIE image download page.
2. Click the login button in the upper right corner of the page and log in with your Huawei account. (If you have not registered, register one first.)
3. On the MindIE image download page, choose the **Image Version** tab. Based on your device form factor, click **Download Now** in the **Operation** column next to the corresponding image.
4. Download the image according to the displayed image download guide, as shown in [Figure 1](#figure1).

    **Figure 1** Image download<a id="figure1"></a>

    ![](../../../figures/image_download.png)

## Using an Image

1. Depending on the device form factor, run the corresponding command below to start the container. The container startup command is for reference only. You can modify the command as required. For details about the command parameters, see [Table 1](#table1).

    **Atlas 800I A2/A3 inference server**

     ```bash
     docker run -it -d --net=host --shm-size=500g \  # For multimodal understanding models with high maximum concurrent requests, it is recommended to set `--shm-size` to no less than 500 GB
        --name <container-name> \
        -w /home \
        --device=/dev/davinci0:rwm \
        --device=/dev/davinci1:rwm \
        --device=/dev/davinci2:rwm \
        --device=/dev/davinci3:rwm \
        --device=/dev/davinci_manager:rwm \
        --device=/dev/hisi_hdc:rwm \
        --device=/dev/devmm_svm:rwm \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
        -v /usr/local/dcmi:/usr/local/dcmi:ro \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
        -v /usr/local/sbin/:/usr/local/sbin:ro \
        -v /home/weight:/home/weight:ro \
        {IMAGE_ID} bash
    ```

    > [!NOTE]
    > - `{IMAGE_ID}` is the image ID. After the image is built, run `docker images` to view the ID, then substitute it into the command above.
    > - The `--device` option must use the `rwm` mount permission, not the more restrictive `rw` or `r`. Reasons:
      >   - On the Atlas 800I A2 inference server, using `rw` allows container entry and `npu-smi` queries, and MindIE services can run normally. However, if another task is already using the mounted NPU (e.g., `npu0` corresponds to `davinci0`), `npu-smi` will print an error and MindIE tasks will fail (at which point `torch.npu.set_device()` will also fail).
      >   - On the Atlas 800I A3 SuperPoD Server, using `rw` causes `npu-smi` to print an error upon container entry and prevents MindIE tasks from running (`torch.npu.set_device()` will fail).

    **Atlas 200I Pro Accelerator Module**

    When starting a container, you must mount the driver libraries and configuration files that `npu-smi` depends on. Failure to do so may cause `npu-smi` commands to fail inside the container. Startup commands vary depending on the container image OS, as shown below.

    - Ubuntu 24.04

        ```bash
        docker run -it -d --net=host --shm-size=100g \  # For multimodal understanding models with high maximum concurrency, it is recommended to set --shm-size to no less than 100 GB
           --name <container-name> \
           --device=/dev/davinci0:/dev/davinci0 \
           --device=/dev/davinci_manager \
           --device=/dev/ascend_manager \
           --device=/dev/user_config \
           -v /etc/sys_version.conf:/etc/sys_version.conf \
           -v /etc/ld.so.conf.d/mind_so.conf:/etc/ld.so.conf.d/mind_so.conf \
           -v /etc/hdcBasic.cfg:/etc/hdcBasic.cfg \
           -v /var/dmp_daemon:/var/dmp_daemon \
           -v /usr/lib64/libmmpa.so:/usr/lib64/libmmpa.so \
           -v /usr/lib64/libcrypto.so.1.1:/usr/lib64/libcrypto.so.1.1 \
           -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
           -v /usr/lib64/libstackcore.so:/usr/lib64/libstackcore.so \
           -v /usr/lib/aarch64-linux-gnu/libyaml-0.so.2:/usr/lib64/libyaml-0.so.2 \
           -v /etc/slog.conf:/etc/slog.conf \
           -v /var/slogd:/var/slogd \
           -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
           -v /path-to-weights:/path-to-weights:ro \
           mindie:3.0.0-300I-DUO-py311-Ubuntu24.04-lts bash
        ```

    - openEuler 24.03:

        ```bash
        docker run -it -d --net=host --shm-size=100g \  # For multimodal understanding models with high maximum concurrency, it is recommended to set --shm-size to no less than 100 GB
           --name <container-name> \
           --device=/dev/davinci0:/dev/davinci0 \
           --device=/dev/davinci_manager \
           --device=/dev/ascend_manager \
           --device=/dev/user_config \
           -v /etc/sys_version.conf:/etc/sys_version.conf \
           -v /etc/ld.so.conf.d/mind_so.conf:/etc/ld.so.conf.d/mind_so.conf \
           -v /etc/hdcBasic.cfg:/etc/hdcBasic.cfg \
           -v /var/dmp_daemon:/var/dmp_daemon \
           -v /usr/lib64/libsemanage.so.2:/usr/lib64/libsemanage.so.2 \
           -v /usr/lib64/libmmpa.so:/usr/lib64/libmmpa.so \
           -v /usr/lib64/libcrypto.so.1.1:/usr/lib64/libcrypto.so.1.1 \
           -v /usr/lib64/libyaml-0.so.2.0.9:/usr/lib64/libyaml-0.so.2 \
           -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
           -v /usr/lib64/libstackcore.so:/usr/lib64/libstackcore.so \
           -v /etc/slog.conf:/etc/slog.conf \
           -v /var/slogd:/var/slogd \
           -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
           -v /path-to-weights:/path-to-weights:ro \
           mindie:3.0.0-300I-DUO-py311-openEuler24.03-lts bash
        ```

    **Table 1** Parameter description <a id="table1"></a>

    |Parameter|Description|
    |----|----|
    |--pids-limit -1|Removes the limit on the number of processes.<br>When using the Atlas 800I A2 inference server with Alibaba Cloud Linux 3.2104 U10, include this parameter in the container start command to disable the process limit.|
    |-it|Starts an interactive terminal (`-i`) and allocates a pseudo-TTY (`-t`), allowing you to interact with the container—e.g., run command-line operations.|
    |-d|Runs the container in the background (detached mode). It does not block the current terminal, allowing you to continue other operations after the container starts.|
    |--net|Makes the container use the host's network stack, giving it direct access to the host's network interfaces. This is suitable for low-latency scenarios where direct access to network resources is required.|
    |--shm-size|Specifies the shared memory size (`/dev/shm`) for the container. Users can set a custom value—`500g` is an example. For multimodal understanding models with high maximum concurrency, it is recommended to set `--shm-size` to no less than 500 GB. <br>The value cannot exceed the remaining physical memory of the host. You can run the `free -h` command to view the remaining physical memory. When data parallelism is enabled (DP > 1), the size of the shared memory needs to be adjusted as the DP value increases.<br>For a DP value of 2, set `shm-size` to at least 2 GB.<br>For a DP value of 4, set `shm-size` to at least 3 GB.<br>For a DP value of 8, set `shm-size` to at least 5 GB.<br>For a DP value of 16, set `shm-size` to at least 9 GB.|
    |--name|Specifies a name for the container. `container-name` is a unique identifier for a container within the current system. You can set it manually. If this parameter is not set, Docker automatically allocates a random name.|
    |--device|Maps the host device to the container. Each `--device` parameter shares a host device (e.g., hardware accelerator or other hardware) directly with the container.<br>`/dev/davinci_manager`: Da Vinci-related management device.<br>`/dev/hisi_hdc`: HDC-related management device.<br>`/dev/devmm_svm`: Memory-related management device<br>/dev/ascend_manager: devices for Ascend device management <br>`/dev/user_config`: a device related to user configuration. It must be mounted when running commands such as `npu-smi` inside containers on the Atlas 200I Pro accelerator module.<br>`/dev/davinciX`: NPU device. `X` indicates the ID, for example, `davinci0`.<br>Run `ll /dev/ \| grep davinci` to check the number and names of devices. Then bind the required devices by modifying `--device=****`.|
    |-v|Maps the folders or directories of the physical machine to the corresponding paths in the container and sets the directories to read-only using the `ro` parameter.<br>`/usr/local/Ascend/driver` contains hardware driver files. The driver must be installed on the host and mapped into the container for usage. Change them according to the actual paths of the driver.<br>`/usr/local/sbin` contains NPU status check commands such as `npu-smi`. Adjust the path based on your environment.<br>`/path-to-weights` points to the directory containing the model weights for container access. Update this path as needed. (The weight file and dataset file are stored in this path.) <br> For the Atlas 200I Pro accelerator module, additional files or directories required by `npu-smi` must be mounted based on the container image OS: <br> `/etc/sys_version.conf` – system version configuration file. <br> `/etc/ld.so.conf.d/mind_so.conf` – dynamic library search path configuration. <br> `/etc/hdcBasic.cfg` – hdc base configuration file. <br> `/var/dmp_daemon` – dmp_daemon runtime directory. <br> `/usr/local/sbin/npu-smi` – `npu-smi` command binary. <br> `/usr/local/Ascend/driver/lib64` – driver dynamic library directory. <br> `/etc/slog.conf`, `/var/slogd` – logging configuration and runtime directory. <br> `/usr/lib64/libmmpa.so`, `/usr/lib64/libcrypto.so.1.1`, `/usr/lib64/libstackcore.so` – common dynamic libraries required by `npu-smi`. <br> For `npu-smi` to function correctly, the following dynamic library dependencies must be present in the container image: <br> `/usr/lib/aarch64-linux-gnu/libyaml-0.so.2` – required for Ubuntu 24.04 images.  <br> `/usr/lib64/libyaml-0.so.2.0.9` – required for openEuler 24.03 images.  <br> `/usr/lib64/libsemanage.so.2` – required for openEuler 24.03 images.|

2. Access the container.

    ```bash
    docker exec -it <container-name> bash
    ```

3. Install the dependency.

    Before running inference with a model, install its dependencies. The path of the dependency installation file (requirements_xxx_.txt) for each model is as follows: `/usr/local/Ascend/atb-models/requirements/models`. To install dependencies for Llama 3 series models (example), run the following commands:

    ```bash
    cd /usr/local/Ascend/atb-models/requirements/models
    pip3 install -r requirements_llama3.txt
    ```

4. Run the following command to enable MindIE log printing:

    ```bash
    export MINDIE_LOG_TO_STDOUT="true"
    ```

5. Use the model for inference.

    For the Llama3 series models, refer to `$ATB_SPEED_HOME_PATH/examples/models/llama3/README.md` in the container. For other models, see [model list](https://www.hiascend.com/software/mindie/modellist).

    Run the following commands to perform inference:

    ```bash
    cd $ATB_SPEED_HOME_PATH
    python examples/run_pa.py --model_path /path-to-weights # Change the weight path
    ```

    The default question and inference result are printed, as shown in the following:

    ```text
    2024-11-18 11:08:13,291 [INFO] [pid: 389497] logging.py-180: Question[0]: What's deep learning?
    2024-11-18 11:08:13,291 [INFO] [pid: 389497] logging.py-180: Answer[0]:  Deep learning is a subset of machine learning that uses neural networks to learn from data. Neural networks are
    2024-11-18 11:08:13,291 [INFO] [pid: 389497] logging.py-180: Generate[0] token num: (0, 20)
    ```

    If you want to customize an input question, set `--input_texts`. For example:

    ```bash
    python examples/run_pa.py --model_path /path-to-weights --input_texts "What is deep learning?"  # Change the weight path
    ```

    > [!NOTE]
    > `$ATB_SPEED_HOME_PATH` has been set in the .bashrc file. You do not need to modify it.

6. MindIE Motor CPP is an inference serving framework designed for general-purpose models, establishing an adaptable and open inference service structure. It interfaces with prominent industry inference frameworks, meeting the high-performance inference needs of LLMs.
   For details about how to connect to Motor CPP, see [Quick Start](https://gitcode.com/Ascend/MindIE-LLM/blob/v3.1.0/docs/zh/user_guide/quick_start/quick_start.md#%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86).
