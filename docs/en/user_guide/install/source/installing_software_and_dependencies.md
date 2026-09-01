# Installing Software Packages and Dependencies

The following describes the software packages and dependencies required for installing MindIE.

## Installing CANN

Install the NPU driver and firmware, and CANN software (Toolkit, ops, and NNAL) of the required version, and configure CANN environment variables. For details, see [CANN Software Installation](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=local).

CANN provides a script for process-level environment variable setting. In training or inference scenarios, you need to invoke this script before using NPUs to execute service code. Otherwise, the service code fails to be executed.

   ```shell
   source /usr/local/Ascend/cann/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   ```

   In the preceding commands, the default root installation path is used as an example. Replace the path based on the actual path of `set_env.sh`.

## Installing PyTorch and Torch NPU

- If the OS is Ubuntu 22.04, install TorchNPU 2.1.0. If the OS is Ubuntu 24.04 LTS, install TorchNPU 2.9.0.
- Install the PyTorch framework and TorchNPU plugin by referring to section "Installing PyTorch" in [TorchNPU Software Installation Guide](https://www.hiascend.com/document/detail/en/Pytorch/730/configandinstg/instg/docs/en/installation_guide/installation_via_binary_package.md).

MindIE components depend on PyTorch and TorchNPU. Refer to the following table to install them as required.

**Table 1** Dependency of MindIE components on the PyTorch framework and TorchNPU plugin

|Component|PyTorch Required or Not|TorchNPU Required or Not|
|--|--|--|
|MindIE Motor CPP|Required|Required|
|MindIE LLM|Required|Required|
|MindIE SD|Required|Required|

> **Note**: If Python 3.10 is used for compilation, torch 2.9.0 and TorchNPU 2.9.0 must be used.
Failure to use these exact versions may result in a missing \_bz2 module and lead to compilation errors.

## Installing ATB models

### Using the `.whl` package

In the root directory where the ATB Models `.whl` package is stored, run the following command to install the package:

```bash
pip install atb_llm-<version>-cp<xxx>-cp<xxx>-linux_<arch>.whl
```

### Using the `.run` package

For the `.run` package installation, ATB Models does not provide an independent software package. Therefore, you need to obtain the package from the MindIE image.

1. In the Ascend image repository, download the image according to the guide. For details, see steps 1 to 4 in [Obtaining the MindIE Image](./image_usage_guide.md#obtaining-the-mindie-image) in "Image Installation".

2. Create a decompression directory (for example, /home/{User name}/Package).

   ```bash
    mkdir /home/{User name}/Package
   ```

3. Grant the read and write permissions on the path.

    ```bash
    chmod u+rw /home/{User name}/Package
    ```

4. Upload the obtained ATB Models software package **Ascend-mindie-atb-models_{version}_linux-{arch}_pyxxx_torchx.x.x-{abi}.tar.gz** to the directory. The ATB Models software package is stored in the ```/opt/package``` directory of the MindIE image package.

    > [!NOTE]
    > The ABI version of ATB Models must be the same as that used during PyTorch compilation. You can call the `torch.compiled_with_cxx11_abi()` API to view the ABI version.
    >
    > - If False is returned, set abi=0.
    > - If True is returned, set abi=1.

5. Go to the directory where the software package is stored and decompress it.

    ```bash
    cd /home/{User name}/Package
    tar -zxf Ascend-mindie-atb-models_{version}_linux-{arch}_pyxxx_torchx.x.x-{abi}.tar.gz
    ```

6. Check the permission on the pip package installation path.
To prevent the error message "module not found" from being displayed after the .whl package is successfully installed, ensure that the current user has the write permission on the installation path of the pip package when pip is used to install the .whl package. You can obtain the installation path of the pip package by running `pip show {Name of the existing package}`. The following is an example:

    ```bash
    pip show pip
    ```

    The following information in bold is the installation path. The actual command output varies according to the actual situation.

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

7. Install the Python package of atb_llm in the Python environment.

    ```bash
    pip install atb_llm-{version}-py3-none-any.whl
    ```

8. Configure environment variable.

    A process-level environment variable setting script is provided to automatically set environment variables. The specified environment variables automatically become invalid after the user process ends.

```bash
source /home/{User name}/Package/set_env.sh
```

   You can also configure permanent environment variables by modifying the ```~/.bashrc``` file. The procedure is as follows:

   1. Run the ```vi ~/.bashrc``` command in any directory as the running user to open the ```.bashrc``` file and append the preceding lines to the file.
   2. Run the ```:wq!``` command to save the file and exit.
   3. Run the ```source ~/.bashrc``` command for the modification to take effect immediately.

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
