# 编译安装指南

## 编译说明

本文档介绍如何从源码编译MindIE-LLM，生成 `.whl` 包，安装与运行。

## 环境准备

### 镜像安装方式

MindIE镜像获取请参见[镜像安装方式](../user_guide/install/source/image_usage_guide.md#获取mindie镜像)。

### 容器/物理机安装方式

1. 容器/物理机安装方式，需要准备的软件包和依赖请参见[准备软件包和依赖](../user_guide/install/source/preparing_software_and_dependencies.md)。
2. 容器/物理机安装方式，软件包和依赖的安装请参见[安装软件包和依赖](../user_guide/install/source/installing_software_and_dependencies.md)。

## 编译安装

1. 安装Python工具。MindIE-LLM 支持 **Python == 3.11**。

    ```bash
    pip install --upgrade pip
    pip install wheel setuptools
    ```

2. 克隆源码仓库。

    ```bash
    git clone https://gitcode.com/Ascend/MindIE-LLM.git
    cd MindIE-LLM
    ```

3. 编译第三方依赖。

    ```bash
    bash build.sh 3rd
    ```

4. 设置环境变量。
    获取 Python site-packages 路径（建议不要强行编码 torch 路径），并配置动态库搜索路径：

    ```bash
    TORCH_PATH=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
    export LD_LIBRARY_PATH=${TORCH_PATH}/lib:${TORCH_PATH}/../torch.libs:$LD_LIBRARY_PATH
    ```

    可选：指定生成 `.whl` 包的版本号：

    ```bash
    export MINDIE_LLM_VERSION_OVERRIDE=3.0.0
    ```
5. 编译生成 MindIE-LLM 的 `.whl` 包。
    在源码根目录下执行：

    ```bash
    pip wheel . --no-build-isolation -v
    ```

    * 编译完成后，会在当前目录生成 `mindie_LLM-<version>-*.whl` 文件。
    * 编译时，`setup.py` 会自动调用 `build.sh` 编译C++代码，并拷贝第三方依赖到包内。
    * 编译后，生成临时目录 `build`、存放二进制的目录 `output` 和 debug 符号表 `llm_debug_symbols` 目录。

6. 安装MindIE-LLM。
   
    ```bash
    old_umask=$(umask)
    umask 027
    pip install mindie_llm*.whl
    umask $old_umask
    ```

7. 编译ATB_Models的 `.whl` 包。

    ```bash
    cd example/atb_models
    pip wheel . --no-build-isolation -v
    ```

8. 安装 ATB_Models。
   
   ```bash
   pip install atb_llm*.whl
   ```

## 升级

升级详情请参见[升级](../user_guide/install/source/upgrade.md)。

## 卸载

卸载详情请参见[卸载](../user_guide/install/source/uninstallation.md)。
