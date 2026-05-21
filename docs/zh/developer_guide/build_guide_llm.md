# 编译安装指南

## 编译说明

本文档介绍使用源码编译MindIE-LLM生成whl软件包，以及如何安装。

## 环境准备

- 需要准备的软件包和依赖请参见[准备软件包和依赖](../user_guide/install/source/preparing_software_and_dependencies.md)。
- 软件包和依赖的安装请参见[安装软件包和依赖](../user_guide/install/source/installing_software_and_dependencies.md)。

## 编译安装

1. 执行以下命令升级pip并安装构建工具。

    ```bash
    pip install --upgrade pip
    pip install wheel setuptools
    ```

2. 执行以下命令克隆源码仓库并进入代码仓目录。

    ```bash
    git clone https://gitcode.com/Ascend/MindIE-LLM.git
    cd MindIE-LLM
    ```

3. 执行以下命令通过设置环境变量用于禁用证书。

    ```bash
    export NO_CHECK_CERTIFICATE=1
    ```

    >[!NOTE]说明
    >使用该命令会禁用证书，可能会导致安全风险，需要注意数据保护。您也可以通过手动下载三方zip包进行上传。

4. （可选）执行以下命令下载unzip工具。

    ```bash
    yum install unzip #openEuler
    ```

5. 执行以下命令编译第三方依赖。

    ```bash
    bash build.sh 3rd
    ```

6. 执行以下命令设置环境变量。
    获取Python site-packages路径（建议不要硬编码torch路径），并配置动态库搜索路径：

    ```bash
    TORCH_PATH=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
    TORCH_NPU_PATH=$(python3 -c "import torch_npu, os; print(os.path.dirname(torch_npu.__file__))")
    export LD_LIBRARY_PATH=${TORCH_PATH}/lib:${TORCH_PATH}/../torch.libs:$LD_LIBRARY_PATH
    export PYTORCH_NPU_INSTALL_PATH=${TORCH_NPU_PATH}
    ```

    可选：指定生成whl软件包的版本号。

    ```bash
    export MINDIE_LLM_VERSION_OVERRIDE=3.0.0
    ```

7. 执行以下命令编译生成MindIE-LLM的whl软件包。
    在源码根目录下执行：

    ```bash
    pip wheel . --no-build-isolation -v
    ```

    * 编译时，`setup.py` 会自动调用 `build.sh` 编译C++代码，并拷贝第三方依赖到包内。
    * 编译完成后，会在当前目录生成 `mindie_llm-<version>-*.whl` 文件。
    * 编译完成后，生成临时目录 `build`、存放二进制的目录 `output` 和 debug 符号表 `llm_debug_symbols` 目录。

8. 执行以下命令安装MindIE-LLM。

    ```bash
    old_umask=$(umask)
    umask 027
    pip install mindie_llm*.whl
    umask $old_umask
    ```

9. 执行以下命令编译ATB_Models的whl软件包。

    ```bash
    cd examples/atb_models
    pip wheel . --no-build-isolation -v
    ```

    > **注意**：使用 Python 3.10 环境编译，需配套 torch 2.9.0 版本 + torch_npu 2.9.0 版本，否则会导致 \_bz2 模块缺失，从而导致编译失败。

10. 执行以下命令安装 ATB_Models。

    ```bash
    pip install atb_llm*.whl
    ```

11. 执行以下命令配置运行环境变量（可选）。

    ATB_Models 安装完成后，需设置其运行所需的环境变量。通过 Python 动态获取 `atb_llm` 的安装路径，以适配不同的 Python 环境和 `site-packages` 位置：

    ```bash
    ATB_LLM_PATH=$(python3 -c "import atb_llm, os; print(os.path.dirname(atb_llm.__file__))")
    export ATB_SPEED_HOME_PATH=${ATB_LLM_PATH}
    export LD_LIBRARY_PATH=${ATB_LLM_PATH}/lib:${LD_LIBRARY_PATH}
    ```

    > [!TIP]提示
    > - 建议将以上命令写入 `~/.bashrc` 或启动脚本中，避免每次手动设置。
    > - MindIE 镜像内置该环境变量，如使用镜像则无需手动设置。
