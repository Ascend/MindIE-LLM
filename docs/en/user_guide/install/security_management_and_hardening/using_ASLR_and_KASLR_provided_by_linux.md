# Using the ASLR and KASLR Functions Provided by Linux

- Address Space Layout Randomization (ASLR) is a security mechanism that enhances vulnerability attack defense capability once enabled.

    The enabling method is as follows:

    Open the `/proc/sys/kernel/randomize_va_space` file and write `2` to the file to enable this function.

- Kernel Address Space Layout Randomization (KASLR) is a security mechanism that  increases the difficulty of attacking kernel vulnerabilities.

    The enabling method is as follows:

  1. View the kernel configuration file.

        ```bash
        vi /boot/config-$(uname -r)
        ```

        If the following information exists, KASLR is supported:

        ```text
        CONFIG_RANDOMIZE_BASE=y
        ```

  2. Open the `/etc/default/grub` configuration file and add `kaslr` to the line where `GRUB_CMDLINE_LINUX_DEFAULT` is located.

        ```bash
        GRUB_CMDLINE_LINUX_DEFAULT="kaslr"
        ```

  3. Update the GRUB configuration.

        ```bash
        sudo update-grub
        ```

  4. Restart the system to enable the KASLR function.

        ```bash
        sudo reboot
        ```
