# Installation Guide

For supported hardware and operating systems, refer to the [Compatibility Checker](https://www.hiascend.com/en/hardware/compatibility).

MindIE LLM and MindIE Motor CPP support installation via container image, offline package, and source code.

The use cases, along with their respective advantages and disadvantages, are outlined below. Choose the installation method that best fits your scenario.

- Image installation: This is the simplest installation method. You can directly download the packaged image from the Ascend community. The image contains necessary dependencies and software such as CANN, PyTorch, and MindIE. You only need to pull the image and start the container. The `.run` package is supported.
- Offline installation: In this method, software such as CANN, PyTorch, and MindIE and dependencies can be installed on physical machines or containers. The `.run` and `.whl` packages can be installed offline.
- Source code installation: To experience the latest functions or modify and enhance the source code, download the code from the repository, compile the code, and install it. WHL package installation is supported for source builds.
  
> [!NOTE]
>
> - The `.whl` package installation is recommended for new users.
> - For existing users upgrading their installation, the `.run` package method is recommended.
