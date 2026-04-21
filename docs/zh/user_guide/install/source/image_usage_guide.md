# 镜像部署方式

指导用户进行MindIE容器镜像安装，请确保服务器能够连接网络。

## 前提条件

- 宿主机已经安装过NPU驱动和固件。如未安装，请参见《CANN 软件安装指南》中的“[选择安装场景](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=openEuler)”（商用版）章节或“[选择安装场景](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=openEuler)”章节（社区版），按如下方式选择安装场景，按“**安装NPU驱动和固件**”章节进行安装。
  - 安装方式：选择“在物理机上安装”。
  - 操作系统：选择使用的操作系统。MindIE支持的操作系统请参考[硬件配套和支持的操作系统](../installation_introduction.md#硬件配套和支持的操作系统)。
  - 安装类型：根据在线或离线的安装方式，选择对应的安装类型。

- 用户在宿主机自行安装Docker（版本要求大于或等于24.x.x）。Docker的安装可参见[安装Docker](../source/docker_installation.md)。
- 配置源之前，请确保安装环境能够连接网络。

## 获取MindIE镜像

1. 单击[昇腾镜像仓库链接](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)，进入MindIE镜像下载页面。
2. 单击页面右上角登录按钮，使用华为账号登录（若未注册，请先注册）。
3. 在MindIE镜像下载页面的“镜像版本”页签，根据您的设备形态，单击对应镜像后方“**操作**”栏中的“立即下载”按钮。
4. 根据弹出的镜像下载操作指导页面下载镜像，如[图1](#figure1)所示。

    **图 1** 镜像下载 <a id="figure1"></a>

    ![](../../../figures/image_download.png)

## 使用镜像

1. 执行以下命令启动容器，容器启动命令仅供参考，可根据需求自行修改，命令参数介绍如[表1](#table1)所示。

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
        mindie:2.3.0-800I-A2-py311-openeuler24.03-lts bash
    ```

    > [!NOTE]说明
    >- “_mindie:2.3.0-800I-A2-py311-openeuler24.03-lts_”为镜像名称和标签，可根据实际情况修改。可在宿主机执行`docker images`命令查看当前机器上已有的镜像。
    >- 对于--device参数，挂载权限设置为rwm，而非权限较小的rw或r，原因如下：
    >- 对于Atlas 800I A2 推理服务器，若设置挂载权限为rw，可以正常进入容器，同时也可以使用npu-smi命令查看npu占用信息，并正常运行MindIE业务；但如果挂载的npu（即对应挂载选项中的davinci_xxx_，如npu0对应davinci0）上有其它任务占用，则使用npu-smi命令会打印报错，且无法运行MindIE任务（此时torch.npu.set\_device\(\)会失败）。
    >- 对于Atlas 800I A3 超节点服务器，若设置挂载权限为rw，进入容器后，使用npu-smi命令会打印报错，且无法运行MindIE任务，此时torch.npu.set_device()会失败。

    **表 1**  参数说明 <a id="table1"></a>

    |参数|说明|
    |----|----|
    |--pids-limit -1|表示解除进程数限制。<br>当Atlas 800I A2 推理服务器使用Alibaba Cloud Linux 3.2104 U10操作系统时，启动容器命令中必须使用该参数解除进程数限制。|
    |-it|表示启动一个交互式终端（-i）并将其连接到容器的标准输入输出 （-t），能够与容器内部进行交互，如运行命令行操作。|
    |-d|表示容器将以后台模式运行，即容器在后台启动。使用该参数后不会阻塞当前终端的操作，可以在启动容器后继续进行其他操作。|
    |--net|表示容器将使用宿主机的网络配置（网络共享），使容器能够直接访问宿主机的网络接口，适用于需要进行低延迟、直接访问网络资源的场景。|
    |--shm-size|表示指定容器的共享内存（/dev/shm）大小，用户可自行设置，1g为示例值。<br>该值不能超过宿主机剩余的物理内存总量，可使用`free -h`命令查看。当开启数据并行（即DP>1时），需要随DP增大调整共享内存大小：<br>当DP=2时，shm-size至少为2g<br>当DP=4时，shm-size至少为3g<br>当DP=8时，shm-size至少为5g<br>当DP=16时，shm-size至少为9g|
    |--name|表示给容器指定一个名称。\<container-name\>是容器的标识符，可以自行设置，且在当前系统中具有唯一性。如果不设置，Docker会自动分配一个随机名称。|
    |--device|表示将宿主机的设备映射到容器内。每个--device参数将宿主机设备（例如硬件加速卡或其他硬件设备）共享给容器，以便容器可以直接访问。<br>/dev/davinci_manager：davinci相关的管理设备。<br>/dev/hisi_hdc：hdc相关管理设备。<br>/dev/devmm_svm：内存管理相关设备。<br>/dev/davinci*X*：NPU设备，*X*是ID号，如：davinci0。<br>可根据`ll /dev/ \| grep davinci`命令查询device个数及名称，根据需要绑定设备，修改上面命令中的"--device=****"。|
    |-v|表示将物理机的文件夹映射到容器内的相应目录，并且使用ro参数将这些目录设置为只读权限。<br>/usr/local/Ascend/driver：该路径包含硬件驱动程序文件，驱动在宿主机上安装，将其映射到容器中，方可在容器中使用。请根据驱动所在实际路径修改。<br>/usr/local/sbin：该路径包含npu-smi等NPU状态查看命令，请根据实际路径修改。<br>/path-to-weights：该路径为设定权重挂载的路径，指向保存权重的目录，使容器能访问权重，请根据实际路径修改。（权重文件和数据集文件同时放置于该路径下）|

2. 执行以下命令进入容器。

    ```bash
    docker exec -it <container-name> bash
    ```

3. 安装依赖。

    使用模型进行推理前需要安装对应的依赖，各模型的依赖安装文件（requirements\__xxx_.txt）所在路径，以Python 3.11为例：/usr/local/lib/python3.11/site-packages/atb_llm/requirements/models。以LLaMA3系列模型为例，使用以下命令安装依赖。

    ```bash
    cd /usr/local/lib/python3.11/site-packages/atb_llm/requirements/models
    pip3 install -r requirements_llama3.txt
    ```

4. 执行以下命令开启MindIE日志打印。

    ```bash
    export MINDIE_LOG_TO_STDOUT="true"
    ```

5. 使用模型进行推理。

    以LLaMA3系列模型为例，具体可参考容器中“$ATB\_SPEED\_HOME\_PATH/examples/models/llama3/README.md”中的说明，其他模型请参见[模型列表](https://www.hiascend.com/software/mindie/modellist)。

    执行以下命令进行推理：

    ```bash
    cd $ATB_SPEED_HOME_PATH
    python examples/run_pa.py --model_path /path-to-weights  # 请修改权重路径
    ```

    打印默认问题“Question”和推理结果“Answer”，如下所示：

    ```text
    2024-11-18 11:08:13,291 [INFO] [pid: 389497] logging.py-180: Question[0]: What's deep learning?
    2024-11-18 11:08:13,291 [INFO] [pid: 389497] logging.py-180: Answer[0]:  Deep learning is a subset of machine learning that uses neural networks to learn from data. Neural networks are
    2024-11-18 11:08:13,291 [INFO] [pid: 389497] logging.py-180: Generate[0] token num: (0, 20)
    ```

    若用户想要自定义输入问题，可使用“--input\_texts”参数设置，如：

    ```bash
    python examples/run_pa.py --model_path /path-to-weights --input_texts "What is deep learning?"  # 请修改权重路径
    ```

    > [!NOTE]说明
    > “$ATB\_SPEED\_HOME\_PATH”已在“.bashrc”中设置好，无需自行设置。

6. MindIE Motor是面向通用模型场景的推理服务化框架，通过开放，可扩展的推理服务化平台架构提供推理服务化能力，支持对接业界主流推理框架接口，满足大语言模型的高性能推理需求。
   对接Motor的方法请参见[快速入门](../../quick_start/quick_start.md#模型推理)。
