# Installing Software Packages and Dependencies

The following describes the software packages and dependencies required for installing MindIE.

## Installing CANN

The CANN packages to be installed include the Toolkit development kit, ops operator package, and NNAL neural network acceleration library.

### Prerequisites

Ensure that the NPU driver and firmware have been installed on the host. If not installed, refer to [Selecting Installation Scenario (Commercial Edition)](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=openEuler) or [Selecting Installation Scenario (Community Edition)](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=openEuler) for in *CANN Software Installation Guide*. Choose the appropriate scenario, then follow the "Installing NPU Driver and Firmware" section to proceed.

- Installation mode: installation on a physical machine
- OS: Select the OS. For details about the OSs supported by MindIE, see [Hardware Requirements and Supported OSs](../installation_introduction.md).
- Installation method: Select the corresponding installation method based on the online or offline installation.

### Installation

Refer to [Selecting Installation Scenario (Commercial Edition)](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=openEuler) or [Selecting Installation Scenario (Community Edition)](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=openEuler) for in *CANN Software Installation Guide*. Choose your scenario as described, read the guide, and then follow the "Installing CANN" section to proceed with the installation.

- Installation mode: installation on a physical machine
- OS: Select the OS. For details about the OSs supported by MindIE, see [Hardware Requirements and Supported OSs](../installation_introduction.md).
- Installation method: Select the corresponding installation method based on the online or offline installation.

## Installing PyTorch and Torch NPU

- If the OS is Ubuntu 22.04, install torch_npu 2.1.0. If the OS is Ubuntu 24.04 LTS, install torch_npu 2.9.0.
- Install the PyTorch framework and torch_npu plugin by referring to section "Installing PyTorch" in [Ascend Extension for PyTorch Software Installation Guide](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md).

MindIE components depend on PyTorch and torch_npu. Refer to the following table to install them as required.

**Table 1** Dependency of MindIE components on the PyTorch framework and torch_npu plugin

|Component|PyTorch Required or Not|torch_npu Required or Not|
|--|--|--|
|MindIE Motor|Required|Required|
|MindIE LLM|Required|Required|
|MindIE SD|Required|Required|

> **Note**: If Python 3.10 is used for compilation, torch 2.9.0 and torch_npu 2.9.0 must be used.
Failure to use these exact versions may result in a missing \_bz2 module and lead to compilation errors.

## Installing ATB models

In the root directory where the ATB Models `.whl` package is stored, run the following command to install the package:

```bash
pip install atb_llm-<version>-cp<xxx>-cp<xxx>-linux_<arch>.whl
```

## Installing Dependencies

### Before You Start

- Install Python and configure the `pip` source in advance.
- You are advised to run the `pip3 install --upgrade pip` command to upgrade `pip` (the `pip` version must be 24.0 or later) to prevent installation failures.

## Installation Procedure

1. Run the following command to install `tritonclient[all]`:

    ```bash
    pip3 install tritonclient[all]
    ```

2. You need to prepare the dependency installation file `requirements.txt`. An example is as follows:

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

3. Start the installation. When installing as a non-root user, append `--user` to the installation command. Run the command in the directory containing `requirements.txt`.

    ```bash
    pip3 install -r requirements.txt
    ```
