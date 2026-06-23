# Build and Installation

## Compilation

This document describes how to compile MindIE-LLM from source code, generate a `.whl` package, and install and run MindIE-LLM.

## Environment Setup

### Image

For details about how to obtain the MindIE image, see [image acquisition](../user_guide/install/source/image_usage_guide.md#obtaining-the-mindie-image).

### Container/Physical Machine

1. For details about the software packages and dependencies required, see [preparing software packages and dependencies](../user_guide/install/source/preparing_software_and_dependencies.md).
2. For details about how to install the software packages and dependencies, see [installing software packages and dependencies](../user_guide/install/source/installing_software_and_dependencies.md).

## Compilation and Installation

1. Install the Python tools. MindIE-LLM supports **Python == 3.10** and **Python == 3.11**.

    ```bash
    pip install --upgrade pip
    pip install wheel setuptools
    ```

2. Clone the source code repository.

    ```bash
    git clone https://gitcode.com/Ascend/MindIE-LLM.git
    cd MindIE-LLM
    ```

3. Compile third-party dependencies.

    ```bash
    bash build.sh 3rd
    ```

4. Set the environment variables.
    Obtain the Python `site-packages` path (you are advised not to hardcode the torch path) and configure the search path for dynamic libraries.

    ```bash
    TORCH_PATH=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
    TORCH_NPU_PATH=$(python3 -c "import torch_npu, os; print(os.path.dirname(torch_npu.__file__))")
    export LD_LIBRARY_PATH=${TORCH_PATH}/lib:${TORCH_PATH}/../torch.libs:$LD_LIBRARY_PATH
    export PYTORCH_NPU_INSTALL_PATH=${TORCH_NPU_PATH}
    ```

    (Optional) Specify the version number of the generated `.whl` package.

    ```bash
    export MINDIE_LLM_VERSION_OVERRIDE=3.0.0
    ```

5. Compile the generated `.whl` package of MindIE-LLM.
    Run the following command in the root directory of the source code:

    ```bash
    pip wheel . --no-build-isolation -v
    ```

    * After the compilation is complete, the `mindie_llm-<version>-*.whl` file is generated in the current directory.
    * During compilation, `setup.py` automatically invokes `build.sh` to compile C++ code and copies third-party dependencies to the package.
    * After the compilation, the temporary directory `build`, the directory `output` for storing binaries, and the debug symbol table directory `llm_debug_symbols` are generated.

6. Install MindIE-LLM.

    ```bash
    old_umask=$(umask)
    umask 027
    pip install mindie_llm*.whl
    umask $old_umask
    ```

7. Compile the `.whl` package of ATB_Models.

    ```bash
    cd examples/atb_models
    pip wheel . --no-build-isolation -v
    ```

    > **Note**: If Python 3.10 is used for compilation, torch 2.9.0 and torch_npu 2.9.0 must be used.
Failure to use these exact versions may result in a missing `_bz2` module and lead to compilation errors.

8. Install ATB_Models.

    ```bash
    pip install atb_llm*.whl
    ```

9. (Optional) Set environment variables.

    After the installation is complete, set environment variables. Use Python to dynamically obtain the `atb_llm` installation path to adapt to different Python environments and `site-packages` locations.

    ```bash
    ATB_LLM_PATH=$(python3 -c "import atb_llm, os; print(os.path.dirname(atb_llm.__file__))")
    export ATB_SPEED_HOME_PATH=${ATB_LLM_PATH}
    export LD_LIBRARY_PATH=${ATB_LLM_PATH}/lib:${LD_LIBRARY_PATH}
    ```

    > [!TIP]Tips
    > - You are advised to write the preceding commands into the `~/.bashrc` or startup script to avoid manual setting each time.
    > - This environment variable is built in the MindIE image. If you use the image, you do not need to manually set it.

## Upgrade

For details about the upgrade, see [upgrade](../user_guide/install/source/upgrade.md).

## Uninstallation

For details about the uninstallation, see [uninstallation](../user_guide/install/source/uninstallation.md).
