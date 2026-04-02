# 性能调优

可通过开启CPU高性能模式、透明大页和jemalloc优化来提升性能，这三种方式相互独立，可以开启其中一个或多个。

> [!NOTE]说明 
> 92核服务器在处理低并发长序列任务时，易发CPU高负载，致使CPU成为系统瓶颈，并引发TPOP性能波动与劣化。建议参照本章节中的方式进行优化。

## 开启CPU高性能模式和透明大页

在裸机中执行以下命令开启CPU高性能模式和透明大页，开启后可提升性能。

-   开启CPU高性能模式，在相同时延约束下，TPS会有约3%的提升。

    ```
    cpupower -c all frequency-set -g performance
    ```

-   开启透明大页，多次实验的吞吐率结果会更稳定。

    ```
    echo always > /sys/kernel/mm/transparent_hugepage/enabled
    ```

    > [!NOTE]说明 
    > 服务化进程可能与模型执行进程抢占CPU资源，导致性能时延波动；可以在启动服务时将服务化进程手动绑核至CPU奇数核，以减少CPU抢占影响，降低性能波动，具体方法如下所示。
    >1.  使用`lscpu`命令查看系统CPU配置情况。
    >    ```
    >    lscpu
    >    ```
    >    CPU相关配置回显信息如下所示：
    >    ```text
    >    NUMA:                   
    >      NUMA node(s):         8
    >      NUMA node0 CPU(s):    0-23
    >      NUMA node1 CPU(s):    24-47
    >      NUMA node2 CPU(s):    48-71
    >      NUMA node3 CPU(s):    72-95
    >      NUMA node4 CPU(s):    96-119
    >      NUMA node5 CPU(s):    120-143
    >      NUMA node6 CPU(s):    144-167
    >      NUMA node7 CPU(s):    168-191
    >    ```
    >2.  使用`taskset -c`命令将服务化进程绑核至CPU奇数核并启动。
    >    ```
    >    taskset -c $cpus ./bin/mindieservice_daemon
    >    ```
    >    $cpus：为CPU配置回显信息中node1、node3、node5或node7的值。

## 开启jemalloc优化

jemalloc优化需要用户自行编译jemalloc动态链接库，并在脚本里引入编译好的动态链接库，具体步骤如下。

1.  单击[链接](https://github.com/jemalloc/jemalloc)下载jemalloc源码，并参考INSTALL.md文件编译安装。
2.  拉起服务前，将jemalloc动态链接库引入环境，执行如下命令。

    ```
    export LD_PRELOAD="{$path_to_lib}/libjemalloc.so:$LD_PRELOAD"
    ```

    其中path\_to\_lib为libjemalloc.so所在路径。

