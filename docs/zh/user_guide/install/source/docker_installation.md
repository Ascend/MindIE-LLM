# 安装Docker

本章以ubuntu 22.04安装docker为例。

1. 确认系统

   ```
   cat /etc/os-release
   ```

   ubuntu系统显示结果：
   ```
   PRETTY_NAME="Ubuntu 22.04 LTS"
   NAME="Ubuntu"
   VERSION_ID="22.04"
   VERSION="22.04 (Jammy Jellyfish)"
   VERSION_CODENAME=jammy
   ID=ubuntu
   ID_LIKE=debian
   HOME_URL="https://www.ubuntu.com/"
   SUPPORT_URL="https://help.ubuntu.com/"
   BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
   PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
   UBUNTU_CODENAME=jammy
   ```

   关注**NAME**、**ID**等参数，确认是否为Ubuntu系统。
2. ubuntu系统安装docker app：

   - 切换可用源：

   ```
   sudo mv /etc/apt/sources.list.d/kubernetes.list /etc/apt/sources.list.d/kubernetes.list.disabled && sudo apt update
   ```

   - 成功显示：

   ```
   Get:1 http://mirrors.tools.huawei.com/ubuntu-ports jammy InRelease [270 kB]
   Hit:2 http://mirrors.tools.huawei.com/ubuntu-ports jammy-updates InRelease
   Hit:3 http://mirrors.tools.huawei.com/ubuntu-ports jammy-backports InRelease
   Fetched 270 kB in 0s (560 kB/s)
   Reading package lists... Done
   Building dependency tree... Done
   Reading state information... Done
   381 packages can be upgraded. Run 'apt list --upgradable' to see them.
   ```

   - 安装docker：

   ```
   sudo apt install docker.io -y
   ```

   - 安装成功显示结果：

   ```
   Reading package lists... Done
   Building dependency tree... Done
   Reading state information... Done
   The following package was automatically installed and is no longer required:
     libjs-highlight.js
   Use 'sudo apt autoremove' to remove it.
   Suggested packages:
     aufs-tools cgroupfs-mount | cgroup-lite debootstrap docker-buildx docker-compose-v2 docker-doc rinse zfs-fuse | zfsutils
   The following packages will be upgraded:
     docker.io
   1 upgraded, 0 newly installed, 0 to remove and 380 not upgraded.
   Need to get 25.6 MB of archives.
   After this operation, 6,515 kB of additional disk space will be used.
   Get:1 http://mirrors.tools.huawei.com/ubuntu-ports jammy-updates/universe arm64 docker.io arm64 28.2.2-0ubuntu1~22.04.1 [25.6 MB]
   Fetched 25.6 MB in 0s (57.3 MB/s)
   Preconfiguring packages ...
   (Reading database ... 166464 files and directories currently installed.)
   Preparing to unpack .../docker.io_28.2.2-0ubuntu1~22.04.1_arm64.deb ...
   Unpacking docker.io (28.2.2-0ubuntu1~22.04.1) over (26.1.3-0ubuntu1~22.04.1) ...
   Setting up docker.io (28.2.2-0ubuntu1~22.04.1) ...
   Warning: The unit file, source configuration file or drop-ins of docker.service changed on disk. Run 'systemctl daemon-reload' to reload units.
   Processing triggers for man-db (2.10.2-1) ...
   Scanning processes...
   Scanning processor microcode...
   Scanning linux images...
   
   Running kernel seems to be up-to-date.
   
   Failed to check for processor microcode upgrades.
   
   No services need to be restarted.
   
   No containers need to be restarted.
   
   No user sessions are running outdated binaries.
   
   No VM guests are running outdated hypervisor (qemu) binaries on this host.
   ```

   - 查看升级docker app版本

   ```
   # 查看docker版本
   docker --version
   
   # 升级到最新版本
   sudo apt update
   sudo apt upgrade docker-ce docker-ce-cli containerd.io
   ```