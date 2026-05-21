# 安装软件包和依赖

介绍安装MindIE前，需要安装的相关软件包和依赖。

## 安装CANN

安装配套版本的NPU驱动固件、CANN软件（Toolkit、ops和NNAL）并配置CANN环境变量，具体请参考《[CANN 软件安装](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0000.html)》（商用版）或《[CANN 软件安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0000.html)》（社区版）。

CANN软件提供进程级环境变量设置脚本，训练或推理场景下使用NPU执行业务代码前需要调用该脚本，否则业务代码将无法执行。

   ```shell
   source /usr/local/Ascend/cann/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   ```

   以上命令以root用户安装后的默认路径为例，请用户根据set_env.sh的实际路径进行替换。

## 安装Pytorch和Torch NPU

- 如果操作系统是ubuntu 22.04，请安装torch_npu 2.1.0；如果操作系统是ubuntu 24.04 LTS，请安装torch_npu 2.9.0。
- 请参考《Ascend Extension for PyTorch 软件安装指南》中的“[安装PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)”章节，安装PyTorch框架和torch_npu插件。

MindIE中各组件依赖PyTorch框架和torch_npu插件，依赖情况如下表所示，请用户依据实际使用需求安装。

**表 1** MindIE各组件依赖PyTorch框架和torch_npu插件说明表

|组件名称|是否需要安装PyTorch框架|是否需要安装torch_npu插件|
|--|--|--|
|MindIE Motor|**必装**|**必装**|
|MindIE LLM|**必装**|**必装**|
|MindIE SD|**必装**|**必装**|

> **注意**：使用 Python 3.10 环境编译，需配套 torch 2.9.0 版本 + torch_npu 2.9.0 版本,
否则会导致 \_bz2 模块缺失，从而导致编译失败。

## 安装ATB Models

### whl包方式

在ATB Models whl包所在根目录，执行如下命令安装：

```bash
pip install atb_llm-<version>-cp<xxx>-cp<xxx>-linux_<arch>.whl
```

### run包方式

对于run包安装方式，由于ATB Models目前未提供单独的软件包，所以需自行从MindIE镜像中获取。

1. 在昇腾镜像仓库，按照指导完成镜像下载。具体操作可参见**镜像安装方式**章节[获取MindIE镜像](./image_usage_guide.md#获取mindie镜像)的步骤1~步骤4。
2. 使用以下命令在环境上新建解压目录（例如：/home/_\{用户名\}_/Package）。

    ```bash
    mkdir /home/{用户名}/Package
    ```

3. 使用以下命令赋予该路径读写权限。

    ```bash
    chmod u+rw /home/{用户名}/Package
    ```

4. 将获取的ATB Models软件包Ascend-mindie-atb-models\__\{version\}_\_linux-_\{arch\}\_py_xxx\__torch_x.x.x__-_\{abi\}_.tar.gz上传至该目录，ATB Models软件包存在于MindIE镜像包的/opt/package目录中。

    >[!NOTE]说明
    >ATB Models的abi版本需要根据环境中安装的PyTorch环境来选择，其版本需要与PyTorch编译时使用的abi版本保持一致，调用torch.compiled\_with\_cxx11\_abi\(\)接口可以查看使用的abi版本：
    >- 如果返回False，则选择abi=0；
    >- 如果返回True，则选择abi=1。

5. 使用以下命令进入软件包所在路径并解压软件包。

    ```bash
    cd /home/{用户名}/Package
    tar -zxf Ascend-mindie-atb-models_{version}_linux-{arch}_pyxxx_torchx.x.x-{abi}.tar.gz
    ```

6. 检查pip包安装路径权限。

    为避免whl包安装成功后，在使用中出现“module not found”错误。使用pip安装whl包时，需要保证当前用户对pip包安装位置拥有写权限，pip包安装路径可以通过`**pip show **\{已存在包的包名\}`方式获得，示例如下。

    ```bash
    pip show pip
    ```

    其安装路径如以下加粗内容所示（具体回显根据实际情况所示）：

    ```text
    Name: pip
    Version: 25.1
    Summary: The PyPA recommended tool for installing Python packages.
    Home-page: https://pip.pypa.io/
    Author:
    Author-email: The pip developers <distutils-sig@python.org>
    License: MIT
    Location: /root/miniconda3/envs/infor/lib/python3.11/site-packages
    Requires:
    Required-by:
    ```

7. 使用以下命令在Python环境中安装atb_llm的Python包。

    ```bash
    pip install atb_llm-{version}-py3-none-any.whl
    ```

8. 配置环境变量。
   当前提供进程级环境变量设置脚本，供用户在进程中引用，以自动完成环境变量设置。用户进程结束后自动失效。

    ```bash
    source /home/{用户名}/Package/set_env.sh
    ```

    用户也可以通过修改\~/.bashrc文件的方式设置永久环境变量，操作如下：

   a. 以运行用户在任意目录下执行`vi \~/.bashrc`命令，打开**.bashrc**文件，在文件最后一行后面添加上述内容。
   b. 执行`:wq!`命令保存文件并退出。
   c. 执行`source \~/.bashrc`命令使其立即生效。

## 安装依赖

### 安装前必读

- 请提前安装Python并配置好pip源。
- 建议执行命令`pip3 install --upgrade pip`进行升级（pip版本需大于或等于24.0），避免因pip版本过低导致安装失败。

## 安装步骤

1. 首先使用以下命令单独安装tritonclient[all]依赖。

    ```bash
    pip3 install tritonclient[all]
    ```

2. 请用户自行准备依赖安装文件requirements.txt，样例如下所示。

    ```text
    gevent==22.10.2
    python-rapidjson>=1.6
    geventhttpclient==2.0.11
    urllib3>=2.1.0
    greenlet==3.0.3
    zope.event==5.0
    zope.interface==6.1
    prettytable~=3.5.0
    jsonschema~=4.21.1
    jsonlines~=4.0.0
    thefuzz~=0.22.1
    pyarrow~=15.0.0
    pydantic~=2.6.3
    sacrebleu~=2.4.2
    rouge_score~=0.1.2
    pillow~=10.3.0
    requests~=2.31.0
    matplotlib>=1.3.0
    text_generation~=0.7.0
    numpy~=1.26.3
    pandas~=2.1.4
    transformers~=4.39.3
    numba==0.61.2
    posix_ipc==1.2.0
    fastapi==0.115.11
    uvicorn==0.34.3
    pybind11==3.0.1
    ```

3. 执行以下命令进行安装。以下命令如果使用非root用户安装，需要在安装命令后加上`--user`，安装命令需在`requirements.txt`所在目录执行。

    ```bash
    pip3 install -r requirements.txt
    ```
