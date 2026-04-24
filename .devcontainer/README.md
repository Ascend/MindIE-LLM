# DevContainer 分步指南

本指南旨在解决“在我的机器上快速搭建构建环境，并正常运行”问题，帮助你实现一个隔离且可复现的开发环境。下面，我们将按步骤完成该开发容器的搭建。

## 步骤1：安装必须的软件与相关插件

- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- [Visual Studio Code](https://code.visualstudio.com/)
- [Visual Studio 扩展插件Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

## 启动

1. 克隆代码仓

    ```bash
    git clone MindIE-LLM代码仓地址
    cd MindIE-LLM
    ```

2. 用VS Code打开项目

    ```Shell
    code .
    ```

3. 按 `F1` 或 `Ctrl + Shift + P`， 输入 `Dev Containers: Reopen in Container`

4. 选择配置方式

5. 容器启动，等待启动完成

## 构建

1. 配置构建并发数，以cpu环境为例，打开`cpu\.devcontainer.json`
2. 找到容器环境变量配置项`containerEnv`, 依据机器情况配置`MAX_COMPILE_CORE_NUM`，默认为2
3. 参见[编译安装指南](../docs/zh/developer_guide/build_guide.md)，进行构建
