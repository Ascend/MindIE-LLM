# Performance Tuning

You can enable the CPU high-performance mode, Transparent Huge Pages (THP), and jemalloc optimization to improve performance. The three modes are independent of each other. You can enable one or more of them.

> [!NOTE]
> When a 92-core server processes low-concurrency long-sequence jobs, the CPU load tends to be high. As a result, the CPU becomes a system bottleneck, causing TPOP performance fluctuation and deterioration. You are advised to perform optimization by referring to this section.

## Enabling the CPU High-Performance Mode and THP

Run the following commands on the bare-metal server (BMS) to enable high-performance CPU mode and THP to improve performance.

- Enabling high-performance CPU mode increases TPS by approximately 3% while maintaining the same latency constraint.

     ```bash
     cpupower -c all frequency-set -g performance
     ```

- If transparent huge page is enabled, the throughput is more stable after multiple tests.

     ```bash
     echo always > /sys/kernel/mm/transparent_hugepage/enabled
     ```

     > [!NOTE]
     > The service process may compete with model execution processes for CPU resources, leading to fluctuations in performance and latency. To mitigate the impact of CPU contention, you can manually bind the service process to an odd-numbered CPU core when starting the service. The detailed method is as follows:
     >1. Run the `lscpu` command to check the CPU configuration of the system.
     >
         > ```bash
         > lscpu
         >  ```
>
     > Information similar to the following is displayed:
     >
         > ```text
         > NUMA:                   
         > NUMA node(s):         8
         > NUMA node0 CPU(s):    0-23
         > NUMA node1 CPU(s):    24-47
         > NUMA node2 CPU(s):    48-71
         > NUMA node3 CPU(s):    72-95
         > NUMA node4 CPU(s):    96-119
         > NUMA node5 CPU(s):    120-143
         > NUMA node6 CPU(s):    144-167
         > NUMA node7 CPU(s):    168-191
         > ```
     >
     >2. Run the `taskset -c` command to bind the service process to an odd-numbered CPU core and start the service.
     >
         > ```bash
         > taskset -c $cpus ./bin/mindieservice_daemon
         > ```
     >
     > `$cpus`: The value of `node1`, `node3`, `node5`, or `node7` in the CPU configuration echo.

## Enabling jemalloc Optimization

To optimize jemalloc, you need to compile the jemalloc dynamic link library and import the compiled dynamic link library to the script. The procedure is as follows:

1. Click [here](https://github.com/jemalloc/jemalloc) to download the jemalloc source code, and compile and install the jemalloc by referring to the `INSTALL.md` file.
2. Before starting the service, import the jemalloc dynamic link library to the environment by running the following command:

     ```bash
     export LD_PRELOAD="{$path_to_lib}/libjemalloc.so:$LD_PRELOAD"
     ```

    Where `path_to_lib` is the path to `libjemalloc.so`.
