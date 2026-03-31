# 安装说明

介绍如何快速完成MindIE LLM软件的安装。

## 安装方案

**本文档包含镜像/容器、物理机场景下，安装MindIE软件的方案**，部署架构如[图1](#figure1)所示。各安装方案的使用场景以及优缺点如下所示，请根据自己的使用场景选择合适的安装方式。

- 镜像安装：该方式是最简单的一种安装方式，用户直接从昇腾社区下载已经打包好的镜像，镜像中已经包含了CANN、PTA、MindIE等必要的依赖与软件，用户只需拉取镜像并启动容器即可。
- 物理机安装：该方式是在不使用Docker容器的情况下，将CANN、PTA、MindIE等软件逐个手动安装到物理机上。这种方式将所有软件直接安装到物理机的操作系统中。
- 容器安装：该方式是将CANN、PTA、MindIE等软件逐个安装到容器中，相当于手动创建镜像。这种方式为用户提供了更高的灵活性，用户可以自由选择和指定软件版本，同时每个容器中的软件环境都是独立的。

    **图 1**  安装方案  <a id="figure1"></a>

    ![](./figures/mindie_installation_diagram.png)

## 硬件配套和支持的操作系统

本章节提供软件包支持的操作系统清单，请执行以下命令查询当前操作系统的版本信息，如果查询的操作系统版本不在对应产品列表中，请替换为支持的操作系统。

```bash
uname -m && cat /etc/*release
```

**表 1**  操作系统支持列表

|硬件|操作系统|
|--|--|
|Atlas 800I A2 推理服务器|AArch64：<li>CentOS 7.6</li><li>CTYunOS 23.01</li><li>CULinux 3.0</li><li>Kylin V10 GFB</li><li>Kylin V10 SP2</li><li>Kylin V10 SP3</li><li>Kylin V10 SP3 2403 4.19.90-89.11.v2401</li><li>Kylin V11</li><li>Ubuntu 22.04</li><li>AliOS3</li><li>BCLinux 21.10 U4</li><li>Ubuntu 24.04 LTS</li><li>openEuler 22.03 LTS</li><li>openEuler 24.03 LTS SP1</li><li>openEuler 22.03 LTS SP4</li><li>Alibaba Cloud Linux 3.2104 U10</li>|
|Atlas 300I Duo 推理卡+Atlas 800 推理服务器（型号 3000）|AArch64：<li>BCLinux 21.10</li><li>Debian 10.8</li><li>Kylin V10 SP1</li><li>Kylin V10 SP3 2403 4.19.90-89.11.v2401</li><li>Kylin V11</li><li>Ubuntu 20.04</li><li>Ubuntu 22.04</li><li>UOS20-1020e</li><li>openEuler 24.03 SP1</li><li>openEuler 22.03 LTS SP4</li>|
|Atlas 300I Duo 推理卡+Atlas 800 推理服务器（型号 3010）|X86_64：<li>Ubuntu 22.04</li>|
|Atlas 800I A3 超节点服务器|AArch64：<li>openEuler 22.03</li><li>CULinux 3.0</li><li>Kylin V10 SP3 2403</li><li>Kylin V11</li><li>BCLinux 21.10 U4（内核版本：5.10.0-200.0.0.131.30）</li><li>CTyunOS 3</li>|
