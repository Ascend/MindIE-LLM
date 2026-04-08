# 安装软件包和依赖

介绍安装MindIE前，需要安装的相关软件包和依赖。

## 安装CANN

需要安装的CANN软件包包括：Toolkit开发套件包、ops算子包和NNAL神经网络加速库。

### 前提条件

宿主机已经安装过NPU驱动和固件。如未安装，请参见《CANN 软件安装指南》中的“[选择安装场景](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=openEuler)”章节（商用版）或“[选择安装场景](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=openEuler)”章节（社区版），按如下方式选择安装场景，按“**安装NPU驱动和固件**”章节进行安装。

-   安装方式：选择“在物理机上安装”。
-   操作系统：选择使用的操作系统，MindIE支持的操作系统请参考[硬件配套和支持的操作系统](../installation_introduction.md)。
-   业务场景：选择“训练&推理&开发调试”。

### 安装

请参见《CANN 软件安装指南》中的“[选择安装场景](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=openEuler)”章节（商用版）或“[选择安装场景](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=openEuler)”章节（社区版），并按如下方式选择安装场景，选择完成后单击“开始阅读”，按“**安装CANN（物理机场景） \> 安装CANN软件包**”章节进行安装。

-   安装方式：选择“在物理机上安装”。
-   操作系统：选择使用的操作系统，MindIE支持的操作系统请参考[硬件配套和支持的操作系统](../installation_introduction.md)。
-   业务场景：选择“训练&推理&开发调试”。
  
## 安装Pytorch和Torch NPU

-   如果是操作系统是ubuntu 22.04，请安装torch_npu 2.1.0；如果操作系统是ubuntu 24.04 LTS，请安装torch_npu 2.6.0。
-   请参见《Ascend Extension for PyTorch 软件安装指南》中的“[安装PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)”章节安装PyTorch框架。
-   请参见《Ascend Extension for PyTorch 软件安装指南》中的“（可选）安装扩展模块”章节安装torch\_npu插件。

MindIE中各组件依赖PyTorch框架和torch\_npu插件，依赖情况如下表所示，请用户依据实际使用需求安装。

**表 1** MindIE各组件依赖PyTorch框架和torch\_npu插件说明表

|组件名称|是否需要安装PyTorch框架|是否需要安装torch_npu插件|
|--|--|--|
|MindIE Motor|**必装**|**必装**|
|MindIE LLM|**必装**|**必装**|
|MindIE SD|**必装**|**必装**|

## 安装ATB Models

在ATB Models whl包所在根目录，执行如下命令安装：

```
pip install atb_llm-<*version>*-cp*xxx*-cp*xxx*-linux_<*arch>*.whl
```

## 安装依赖

### 安装前必读

- 请提前安装Python并配置好pip源。
- 建议执行命令`pip3 install --upgrade pip`进行升级（pip版本需大于或等于24.0），避免因pip版本过低导致安装失败。

## 安装步骤

1.  首先使用以下命令单独安装tritonclient\[all\]依赖。

    ```
    pip3 install tritonclient[all]
    ```

2.  请用户自行准备依赖安装文件requirements.txt，样例如下所示。

    ```
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
    ```

3.  执行以下命令进行安装。如下命令如果使用非root用户安装，需要在安装命令后加上**--user**，安装命令需在`requirements.txt`所在目录执行。

    ```
    pip3 install -r requirements.txt
    ```
