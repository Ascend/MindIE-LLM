# （可选）安装systemd-coredump工具

systemd-coredump是由systemd提供的一个核心转储（core dump）处理与收集工具。当进程崩溃（例如收到SIGSEGV、SIGABRT等信号）时，系统内核会尝试生成core dump文件，用于调试分析。而systemd-coredump接管了这一步骤，使核心转储的生成、压缩、存储、记录与访问更加安全、高效、可控。

## 操作步骤

通过在宿主机安装systemd-coredump并保存core dump文件，容器和Pod内无需任何操作。

1. 安装systemd-coredump，OpenEuler操作系统默认已经安装，Ubuntu操作系统执行以下命令进行安装。

    ```bash
    apt install systemd-coredump
    ```

    > [!NOTE]说明
    > 安装systemd-coredump后，可能造成网络共享存储出现“Input/Output Error”报错，重启宿主机后即可修复。

2. 执行以下命令查看systemd-coredump是否安装成功。

    ```bash
    ls -l /usr/lib/systemd/systemd-coredump
    ```

    打印信息中有以下内容则表示安装成功：

    ```text
    /usr/lib/systemd/systemd-coredump
    ```

3. 配置systemd-coredump，执行以下命令打开coredump.conf文件。

    ```bash
    vi /etc/systemd/coredump.conf
    ```

    推荐配置样例如下所示，参数解释如[表1](#table1)所示：

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

    **表 1**  coredump.conf关键参数说明 <a id="table1"></a>

    |参数|取值范围|说明|
    |--|--|--|
    |Storage|<ul><li>none：不保存core dump文件。</li><li>external：将core dump文件保存至磁盘的/var/lib/systemd/coredump/目录，也可以使用coredumpctl命令进行查看。</li><li>journal：将core dump文件不保存到磁盘，仅写入systemd journal，可用coredumpctl命令进行查看。</li><li>both：同时保存到磁盘和journal。</li></ul>|配置core dump的保存位置，默认值：external。|
    |Compress|<ul><li>yes：开启。</li><li>no：不开启。</li></ul>|是否启用压缩功能。启用后，systemd-coredump会压缩core dump文件，默认值：yes。<ul><li>压缩率高达100~300倍（取决于压缩格式）。</li><li>Ubuntu通常以zstd格式压缩，OpenEuler以lz4格式压缩。</li></ul>|
    |ProcessSizeMax|-|允许处理的最大内存字节数。 超过此大小的内存转储有可能会被保存下来， 但是肯定不会生成回溯。<br>同时设置Storage=none与ProcessSizeMax=0时将会禁止处理一切内存转储，同时仅为每个内存转储事件记录一条简略的日志消息。|
    |ExternalSizeMax|-|允许保存的最大内存字节数 (未压缩前)。<br>建议设为300G，MindIE程序实测中最大core dump约为120GB，可根据磁盘大小设置合适值。|
    |JournalSizeMax|-|当参数Storage取值为journal或both时，限制core dump写入systemd journal的大小，当core dump超过设置的值就停着写入systemd journal。<br>当Storage配置为external时，该参数无效。|
    |MaxUse|<ul><li>置空：不限制；（不推荐）</li><li>其他值：单位有K、M和G，如果设置为10G，表示/var/lib/systemd/coredump/目录最多占用10GB。</li></ul>|限制/var/lib/systemd/coredump/目录占用的最大空间，建议设置为10GB。<br>当超过该限制时，core dump文件就会以轮转模式存入。|
    |KeepFree|-|保留磁盘可用的空间阈值。<br>即使未到参数MaxUse设置的值，若磁盘剩余空间低于该值，则core dump文件就会以轮转模式存入。<br>例：KeepFree=2G，表示保证磁盘至少预留2GB的空闲空间。|

4. 执行以下命令使能配置。

    ```bash
    sudo echo "|/usr/lib/systemd/systemd-coredump %P %u %g %s %t %c %e" > /proc/sys/kernel/core_pattern
    sudo systemctl daemon-reexec
    sudo systemctl daemon-reload
    ```

5. 执行以下命令查看core dump的相关信息与调试。
   - 查看已保存的coredump：

        ```bash
        sudo coredumpctl list
        ```

   - 查看core dump的详细信息：

        ```bash
        sudo coredumpctl info <PID>
        ```

   - 导出core dump到文件：

        ```bash
        sudo coredumpctl dump <PID> > /tmp/corefile
        ```

   - 分析core dump：

        ```bash
        gdb /path/to/program /tmp/corefile
        ```
