# 使用Linux自带的ASLR和KASLR功能

- ASLR（Address Space Layout Randomization）是一种安全机制，开启后可以增强漏洞攻击防护能力。

    开启方式如下所示：

    打开“/proc/sys/kernel/randomize\_va\_space”文件并写入“2”，即可开启该功能。

- KASLR（Kernel Address Space Layout Randomization）是一种安全机制，开启后可以增加针对内核漏洞的攻击难度。

    开启方式如下所示：

  1. 使用以下示例命令查看内核配置文件。

        ```bash
        vi /boot/config-$(uname -r)
        ```

        如果存在以下行则表示支持KASLR。

        ```text
        CONFIG_RANDOMIZE_BASE=y
        ```

  2. 打开配置文件/etc/default/grub，在GRUB\_CMDLINE\_LINUX\_DEFAULT所在行的添加kaslr参数，示例如下所示。

        ```json
        GRUB_CMDLINE_LINUX_DEFAULT="kaslr"
        ```

  3. 使用以下命令更新grub配置。

        ```bash
        sudo update-grub
        ```

  4. 使用以下命令重启系统后即开启KASLR功能。

        ```bash
        sudo reboot
        ```
