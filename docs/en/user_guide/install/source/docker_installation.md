# Installing Docker

The following uses Ubuntu 22.04 as an example to describe how to install Docker.

1. Check the OS.

   ```bash
   cat /etc/os-release
   ```

   The following information is displayed for Ubuntu:

   ```text
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

   Check whether the system is Ubuntu by checking the values of parameters such as `NAME` and `ID`.
2. Install Docker on Ubuntu.

   - Switch to the available source.

   ```bash
   sudo mv /etc/apt/sources.list.d/kubernetes.list /etc/apt/sources.list.d/kubernetes.list.disabled && sudo apt update
   ```

   - If the operation is successful, the following information is displayed:

   ```text
   Get:1 http://mirrors.tools.huawei.com/ubuntu-ports jammy InRelease [270 kB]
   Hit:2 http://mirrors.tools.huawei.com/ubuntu-ports jammy-updates InRelease
   Hit:3 http://mirrors.tools.huawei.com/ubuntu-ports jammy-backports InRelease
   Fetched 270 kB in 0s (560 kB/s)
   Reading package lists... Done
   Building dependency tree... Done
   Reading state information... Done
   381 packages can be upgraded. Run 'apt list --upgradable' to see them.
   ```

   - Install Docker.

   ```bash
   sudo apt install docker.io -y
   ```

   - If the installation is successful, the following information is displayed:

   ```text
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

   - Check and upgrade the Docker version.

   ```bash
   # Run the following command to view the Docker version
   docker --version

   # Update Docker to the latest version
   sudo apt update
   sudo apt upgrade docker-ce docker-ce-cli containerd.io
   ```
