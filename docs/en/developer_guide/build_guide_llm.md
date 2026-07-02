# Build and Installation

## Compilation

This document explains how to build the MindIE-LLM `.whl` package from source and install it.

## Environment Setup

- For details about the software packages and dependencies to be prepared, see [Preparing Software Packages and Dependencies](../user_guide/install/source/preparing_software_and_dependencies.md).
- For details about how to install software packages and dependencies, see [Installing Software Packages and Dependencies](../user_guide/install/source/installing_software_and_dependencies.md).

## Compilation and Installation

1. Run the following commands to upgrade pip and install the build tool:

    ```bash
    pip install --upgrade pip
    pip install wheel setuptools
    ```

2. Run the following commands to clone the source code repository and go to the code repository directory:

    ```bash
    git clone https://gitcode.com/Ascend/MindIE-LLM.git
    cd MindIE-LLM
    ```

3. Run the following command to set environment variables to disable the certificate:

    ```bash
    export NO_CHECK_CERTIFICATE=1
    ```

    >[!NOTE]NOTE
    >Running this command will disable the certificate, which may cause security risks. Therefore, pay attention to data protection. You can also manually download a third-party ZIP package and upload it.

4. (Optional) Run the following command to download the unzip tool:

    ```bash
    yum install unzip #openEuler
    ```

5. Run the following command to compile the third-party dependencies:

    ```bash
    bash build.sh 3rd
    ```

6. Set the environment variable.
    Obtain the Python `site-packages` path (do not hardcode the torch path) and configure the dynamic library search path.

    ```bash
    TORCH_PATH=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
    TORCH_NPU_PATH=$(python3 -c "import torch_npu, os; print(os.path.dirname(torch_npu.__file__))")
    export LD_LIBRARY_PATH=${TORCH_PATH}/lib:${TORCH_PATH}/../torch.libs:$LD_LIBRARY_PATH
    export PYTORCH_NPU_INSTALL_PATH=${TORCH_NPU_PATH}
    ```

    (Optional) Specify the version number of the generated `.whl` software package.

    ```bash
    export MINDIE_LLM_VERSION_OVERRIDE=3.0.0
    ```

7. Compile and generate the `.whl` software package of MindIE-LLM.
    Run the following command in the root directory of the source code:

    ```bash
    pip wheel . --no-build-isolation -v
    ```

    * During compilation, `setup.py` automatically invokes `build.sh` to compile C++ code and copies third-party dependencies to the package.
    * After the compilation is complete, the `mindie_llm-<version>-*.whl` file is generated in the current directory.
    * After the compilation is complete, the temporary directory `build`, binary directory `output`, and debug symbol table `llm_debug_symbols` are generated.

8. Run the following command to install MindIE-LLM:

    ```bash
    old_umask=$(umask)
    umask 027
    pip install mindie_llm*.whl
    umask $old_umask
    ```

9. Compile the `.whl` software package of ATB_Models.

    ```bash
    cd examples/atb_models
    pip wheel . --no-build-isolation -v
    ```

    > **Note**: If Python 3.10 is used for compilation, torch 2.9.0 and torch_npu 2.9.0 must be used. Otherwise, the \_bz2 module will be missing, causing compilation failure.

10. Run the following command to install ATB_Models:

    ```bash
    pip install atb_llm*.whl
    ```

11. (Optional) Configure the operating environment variables.

    After the installation is complete, set environment variables. Use Python to dynamically obtain the `atb_llm` installation path to adapt to different Python environments and `site-packages` locations.

    ```bash
    ATB_LLM_PATH=$(python3 -c "import atb_llm, os; print(os.path.dirname(atb_llm.__file__))")
    export ATB_SPEED_HOME_PATH=${ATB_LLM_PATH}
    export LD_LIBRARY_PATH=${ATB_LLM_PATH}/lib:${LD_LIBRARY_PATH}
    ```

    > [!TIP]Tips
    > - You are advised to write the preceding commands into the `~/.bashrc` or startup script to avoid manual setting each time.
    > - This environment variable is built into the MindIE image. If you use the image, you do not need to manually set it.
