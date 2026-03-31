# 容器安装方式

介绍进行MindIE容器化安装，请确保服务器能够连接网络。

## 前提条件

- 宿主机已经安装过NPU驱动和固件。如未安装，请参见《CANN 软件安装指南》中的“[选择安装场景](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=openEuler)”（商用版）章节或“[选择安装场景](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=openEuler)”章节（社区版），按如下方式选择安装场景，按“**安装NPU驱动和固件**”章节进行安装。
    - 安装方式：选择“在物理机上安装”。
    - 操作系统：选择使用的操作系统，MindIE支持的操作系统请参考[硬件配套和支持的操作系统](../installation_introduction.md#硬件配套和支持的操作系统)。
    - 业务场景：选择“训练&推理&开发调试”。

- 宿主机已参见[准备软件包和依赖](../source/preparing_software_and_dependencies.md)章节准备好需要安装的软件包和依赖。
- 用户在宿主机自行安装Docker（版本要求大于或等于24_.x.x_）。Docker的安装可参见[安装Docker](../source/docker_installation.md)。
- 配置源之前，请确保安装环境能够连接网络。  

## 操作步骤

1. 拉取操作系统镜像。

    ```bash
    docker pull ubuntu:22.04
    ```

    此处拉取Ubuntu 22.04仅为示例，用户可拉取其他支持的操作系统版本，但确保镜像拉取的操作系统符合[硬件配套和支持的操作系统](../installation_introduction.md#硬件配套和支持的操作系统)中的要求。

    > [!NOTE]说明
    > torch_npu 2.1.0版本请拉取ubuntu 22.04，torch_npu 2.6.0版本请拉取ubuntu 24.04 LTS。
    > 在一个全新的容器内可能会出现apt源下载路径问题，请用户配置Ubuntu 22.04的专用源，提升下载速度。
    > 安装过程需要下载相关依赖，请确保安装环境能够连接网络。
    > 请在root用户下执行`apt update`命令检查源是否可用。
    > 如果命令执行报错或者后续安装依赖时等待时间过长甚至报错，则检查网络是否连接或者把“/etc/apt/sources.list“文件中的源更换为可用的源或使用镜像源（以配置华为镜像源为例，可参考[华为开源镜像站](https://mirrors.huaweicloud.com/)）。

2. 拉起容器，挂载宿主机目录。在容器安装过程中，用户无需在容器内安装驱动，只需根据不同产品类型将以下示例中的目录挂载至容器内。

    可参考如下示例命令启动容器，具体挂载信息可根据产品路径和实际需求修改。

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

    > [!NOTE]说明
    > 对于--device参数，挂载权限设置为rwm，而非权限较小的rw或r，原因如下：
    >- 对于Atlas 800I A2 推理服务器，若设置挂载权限为rw，可以正常进入容器，同时也可以使用npu-smi命令查看npu占用信息，并正常运行MindIE业务；但如果挂载的npu（即对应挂载选项中的davinci_xxx_，如npu0对应davinci0）上有其它任务占用，则使用npu-smi命令会打印报错，且无法运行MindIE任务（此时torch.npu.set\_device\(\)会失败）。
    >- 对于Atlas 800I A3 超节点服务器，若设置挂载权限为rw，进入容器后，使用npu-smi命令会打印报错，且无法运行MindIE任务（此时torch.npu.set\_device\(\)会失败）。

    **表 1**  参数说明

    |参数|参数说明|
    |--|--|
    |--pids-limit -1|表示解除进程数限制。<br>当Atlas 800I A2 推理服务器使用Alibaba Cloud Linux 3.2104 U10操作系统时，启动容器命令中必须使用该参数解除进程数限制。|
    |--shm-size=1g|表示指定容器的共享内存（/dev/shm）大小，用户可自行设置，1g为示例值。<br>该值不能超过宿主机剩余的物理内存总量，可使用`free -h`命令查看。当开启数据并行（即DP>1）时，需要随DP增大调整共享内存大小：<li>当DP=2时，shm-size至少为2g;</li><li>当DP=4时，shm-size至少为3g;</li><li>当DP=8时，shm-size至少为5g;</li><li>当DP=16时，shm-size至少为9g。</li>|
    |--name|容器名，请根据需要自行设定。|
    |--device|表示映射的设备，可以挂载一个或者多个设备。<br>需要挂载的设备如下：<li>/dev/davinci_manager：davinci相关的管理设备。</li><li>/dev/hisi_hdc：hdc相关管理设备。</li><li>/dev/devmm_svm：内存管理相关设备。</li><li>/dev/davinci0：需要挂载的卡号。</li><br>可根据`ll \/dev\/ \| grep davinci`命令查询device个数及名称方式，根据需要绑定设备，修改上面命令中的"--device=****"。|
    |-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro|将宿主机目录“/usr/local/Ascend/driver ”挂载到容器，请根据驱动所在实际路径修改。|
    |-v /usr/local/sbin:/usr/local/sbin:ro|挂载容器内需要使用的工具。|
    |-v /path-to-weights:/path-to-weights:ro|挂载宿主机模型权重所在目录。|

3. 确认**npu-smi**工具是否成功挂载（默认路径为"/usr/local/sbin/"，请根据实际情况调整路径）。
    1. 使用命令查看该目录下的文件列表，确认**npu-smi**工具存在：

        ```bash
        ll /usr/local/sbin/
        ```

    2. 检查**npu-smi**的权限设置。

        确保**npu-smi**文件具有适当的执行权限。可以通过以下命令更改权限：

        ```bash
        chmod 555 /usr/local/sbin/npu-smi
        ```

    3. 验证执行权限。

        执行**npu-smi info**命令，检查是否有输出信息。如果没有输出信息，请再次检查上述步骤：

        ```bash
        npu-smi info
        ```

4. 进入容器。

    ```bash
    docker exec -it <container-name> /bin/bash
    ```

5. 请将/usr/local/Ascend/driver/下的so文件路径配置到LD\_LIBRARY\_PATH中，如下所示：

    ```bash
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
    ```

6. 请参见[安装软件包和依赖](../source/installing_software_and_dependencies.md)章节在容器中安装依赖、CANN和PyTorch。
7. 请参见[安装MindIE](installation_on_a_physical_machine.md)章节在容器中安装MindIE，安装完成后，即可在容器内进行MindIE的业务部署。
