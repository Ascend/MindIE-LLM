# (Optional) Installing systemd-coredump

systemd-coredump is a core dump processing and collection tool provided by systemd. When a process crashes (e.g., due to SIGSEGV, SIGABRT, etc.), the kernel attempts to generate a core dump for debugging. systemd-coredump takes over this process, making core dump generation, compression, storage, logging, and access more secure, efficient, and manageable.

## Procedure

Install systemd-coredump on the host and save the core dump file. No operation is required in the container or pod.

1. Install systemd-coredump. It is installed by default in openEuler. For Ubuntu, run the following command to install it:

    ```bash
    apt install systemd-coredump
    ```

    > [!NOTE]
    > After systemd-coredump is installed, network shared storage may report an "Input/Output Error". Restarting the host resolves the issue.

2. Check whether the installation is successful.

    ```bash
    ls -l /usr/lib/systemd/systemd-coredump
    ```

    If the following information is displayed, the installation is successful:

    ```text
    /usr/lib/systemd/systemd-coredump
    ```

3. Open the `coredump.conf` file.

    ```bash
    vi /etc/systemd/coredump.conf
    ```

    The following is a recommended configuration example. For details about the parameters, see [Table 1](#table1).

    ```text
    [Coredump]
    Storage=external
    Compress=yes
    ProcessSizeMax=300G
    ExternalSizeMax=300G
    JournalSizeMax=512M
    MaxUse=10G
    KeepFree=2G
    ```

    **Table 1** Key parameters in coredump.conf <a id="table1"></a>

    |Parameter|Value Range|Description|
    |--|--|--|
    |Storage|<ul><li>`none`: Do not save core dump files. </li><li>`external`: Save core dump files to `/var/lib/systemd/coredump/` on drive. Files can also be viewed with the `coredumpctl` command. </li><li>`journal`: Write core dump files only to the systemd journal (not to drive). Files can also be viewed with the `coredumpctl` command. </li><li>`both`: Save core dump files both to drive and to the journal.</li></ul>|Specifies the location for saving the core dump file. The default value is `external`.|
    |Compress|<ul><li>`yes`: enabled. </li><li>`no`: disabled.</li></ul>|Specifies whether to enable the compression function. After this function is enabled, systemd-coredump will compress the core dump file. The default value is `yes`. <ul><li>Compression ratios range from 100× to 300×, depending on the format. </li><li>Ubuntu typically uses zstd compression, while openEuler uses lz4.</li></ul>|
    |ProcessSizeMax|-|Specifies the maximum memory bytes allowed for processing. A dump exceeding this size may still be saved, but a backtrace will not be generated.<br>Setting both `Storage=none` and `ProcessSizeMax=0` disables dump processing entirely, logging only a brief message per dump event.|
    |ExternalSizeMax|-|Specifies the maximum memory bytes allowed to be saved (before compression).<br>A value of `300G` is recommended, as the largest core dump observed in MindIE testing was approximately 120 GB. Adjust this value based on available drive space.|
    |JournalSizeMax|-|When `Storage` is set to `journal` or `both`, this parameter limits the size of core dumps written to the systemd journal. If a core dump exceeds the configured value, writing to the journal stops.<br>This parameter has no effect when `Storage` is set to `external`.|
    |MaxUse|<ul><li>Empty: No limit (not recommended).</li><li>Other values: Supports units K, M, and G. For example, setting `10G` means the `/var/lib/systemd/coredump/` directory can use up to 10 GB.</li></ul>|Limits the maximum drive usage for `/var/lib/systemd/coredump/`. The recommended value is `10G`.<br>Once this limit is exceeded, core dump files will be stored in rotation.|
    |KeepFree|-|Specifies the threshold for reserving free drive space.<br>Even if the `MaxUse` limit has not been reached, core dump files will rotate if the remaining drive space falls below this value.<br>Example: `KeepFree=2G` ensures at least 2 GB of free space is preserved on the drive.|

4. Run the following command to enable the configuration:

    ```bash
    sudo echo "|/usr/lib/systemd/systemd-coredump %P %u %g %s %t %c %e" > /proc/sys/kernel/core_pattern
    sudo systemctl daemon-reexec
    sudo systemctl daemon-reload
    ```

5. Run the following commands to view the core dump information and perform debugging.
   - View the saved core dump:

        ```bash
        sudo coredumpctl list
        ```

   - View the detailed information about the core dump:

        ```bash
        sudo coredumpctl info <PID>
        ```

   - Export the core dump to a file:

        ```bash
        sudo coredumpctl dump <PID> > /tmp/corefile
        ```

   - Analyze the core dump:

        ```bash
        gdb /path/to/program /tmp/corefile
        ```
