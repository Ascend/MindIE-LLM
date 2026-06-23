# Installation Guide

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-05-28T09:07:29.543Z pushedAt=2026-06-02T01:17:57.842Z -->

Describes how to quickly complete the installation of MindIE LLM.

## Installation Methods

This document covers the scenarios for installing MindIE via image, container, and physical machine. The deployment architecture is shown in [Figure 1](#figure1). The use cases, advantages, and disadvantages of each installation method are listed below. Select the appropriate installation method based on your use case.

- Image: This is the simplest installation method. Users can download a ready-to-use image from the Ascend Community, which includes all necessary dependencies and software such as CANN, PyTorch, and MindIE. Simply pull the image and start the container.

- Physical machine: This method involves manually installing software components such as CANN, PyTorch, and MindIE one by one onto a physical machine without using Docker containers. All software is installed directly into the host operating system.

- Container: This method involves installing software components such as CANN, PyTorch, and MindIE one by one into a container, which is essentially manually building an image. It offers greater flexibility, allowing users to freely select and specify software versions while keeping the software environment isolated within each container.

    **Figure 1**  Installation scheme <a id="figure1"></a>

    ![](./figures/mindie_installation_diagram.png)

## Hardware Compatibility and Supported Operating Systems

This section provides the list of operating systems supported by the software package. Run the following command to query the current operating system version. If the queried operating system version is not in the corresponding product list, replace it with a supported operating system.

```bash
uname -m && cat /etc/*release
```

**Table 1**  Supported operating systems

|Hardware|Operating System|
|--|--|
|Atlas 800I A2 Inference Server|AArch64: <li>CentOS 7.6</li><li>CTYunOS 23.01</li><li>CULinux 3.0</li><li>Kylin V10 GFB</li><li>Kylin V10 SP2</li><li>Kylin V10 SP3</li><li>Kylin V10 SP3 2403 4.19.90-89.11.v2401</li><li>Kylin V11</li><li>Ubuntu 22.04</li><li>AliOS3</li><li>BCLinux 21.10 U4</li><li>Ubuntu 24.04 LTS</li><li>openEuler 22.03 LTS</li><li>openEuler 24.03 LTS SP1</li><li>openEuler 22.03 LTS SP4</li><li>Alibaba Cloud Linux 3.2104 U10</li>|
|Atlas 300I Duo Inference Card + Atlas 800 Inference Server (Model 3000)|AArch64: <li>BCLinux 21.10</li><li>Debian 10.8</li><li>Kylin V10 SP1</li><li>Kylin V10 SP3 2403 4.19.90-89.11.v2401</li><li>Kylin V11</li><li>Ubuntu 20.04</li><li>Ubuntu 22.04</li><li>UOS20-1020e</li><li>openEuler 24.03 SP1</li><li>openEuler 22.03 LTS SP4</li>|
|Atlas 300I Duo Inference Card + Atlas 800 Inference Server (Model 3010)|x86_64: <li>Ubuntu 22.04</li>|
|Atlas 800I A3 SuperPoD Server|AArch64: <li>openEuler 22.03</li><li>CULinux 3.0</li><li>Kylin V10 SP3 2403</li><li>Kylin V11</li><li>BCLinux 21.10 U4 (Kernel version: 5.10.0-200.0.0.131.30)</li><li>CTyunOS 3</li>|
