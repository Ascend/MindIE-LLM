# README

## 变量解释
| 变量名      | 含义                                   |
| ----------- | -------------------------------------- |
| working_dir | 加速库及模型库下载后放置的目录         |
| cur_dir     | 运行指令或执行脚本时的路径（当前目录） |
| version     | 版本                                   |

## 环境准备
### 依赖版本
- 模型仓代码配套可运行的硬件型号
  - Atlas 800I A3（64GB显存）
  - Atlas 800I A2（32GB/64GB显存）
  - Atlas 300I DUO（96GB显存）
- 模型仓代码运行相关配套软件
  - 系统OS
  - 驱动（HDK）
  - CANN
  - Python
  - PTA
  - 开源软件依赖
- 版本配套关系
  - 当前模型仓需基于CANN包8.0版本及以上，Python 3.10/3.11，torch 2.1.0/2.6.0进行环境部署与运行

### 1.1 安装HDK

- 详细信息可参见昇腾社区驱动与固件
- 第一次安装时：先安装driver，再安装firmwire，最后执行`reboot`指令重启服务器生效
- 若服务器上已安装驱动固件，进行版本升级时：先安装firmwire，再安装driver，最后执行`reboot`指令重启服务器生效

#### 1.1.1 安装firmwire

- 下载

| 包名                                     | 硬件型号     |
| ---------------------------------------- | ------------ |
| Ascend-hdk-*-npu-firmware_${version}.run | Atlas 800I A2 |
| Ascend-A3-hdk-npu-firmware_${version}.run | Atlas 800I A3 |

  - 根据芯片型号下载对应的安装包

- 安装
  ```bash
  chmod +x Ascend-hdk-*-npu-firmware_${version}.run
  ./Ascend-hdk-*-npu-firmware_${version}.run --full
  ```

#### 1.1.2 安装driver

- 下载

| cpu     | 包名                                                 |
| ------- | ---------------------------------------------------- |
| aarch64 | Ascend-hdk-*-npu-driver_${version}_linux-aarch64.run |
| x86     | Ascend-hdk-*-npu-driver_${version}_linux-x86-64.run  |
  - 根据CPU架构以及npu型号下载对应的driver

- 安装
  ```bash
  chmod +x Ascend-hdk-*-npu-driver_${version}_*.run
  ./Ascend-hdk-*-npu-driver_${version}_*.run --full
  ```

### 1.2 安装CANN

- 详细信息可参见[昇腾社区CANN软件](https://www.hiascend.com/software/cann)
- 安装顺序：先安装toolkit 再安装kernel

#### 1.2.1 安装toolkit

- 下载

| cpu     | 包名（其中`${version}`为实际版本）                 |
| ------- | ------------------------------------------------ |
| aarch64 | Ascend-cann-toolkit_${version}_linux-aarch64.run |
| x86     | Ascend-cann-toolkit_${version}_linux-x86_64.run  |

- 安装
  ```bash
  # 安装toolkit  以arm为例
  chmod +x Ascend-cann-toolkit_${version}_linux-aarch64.run
  ./Ascend-cann-toolkit_${version}_linux-aarch64.run --install
  source /usr/local/Ascend/cann/set_env.sh
  ```

#### 1.2.2 安装ops

- 下载

| 包名（其中`${version}`为实际版本，`${cpu}`为实际cpu架构  |
| ----------------------------------------------------- |
| Ascend-cann-{芯片型号}-ops_${version}_linux-${cpu}.run      |

  - 根据芯片型号选择对应的安装包

- 安装
  ```bash
  chmod +x Ascend-cann-kernels-*_${version}_linux.run
  ./Ascend-cann-kernels-*_${version}_linux.run --install
  source /usr/local/Ascend/cann/set_env.sh
  ```

#### 1.2.3 安装加速库
- 下载加速库

  | 包名（其中`${version}`为实际版本）            |
  | -------------------------------------------- |
  | Ascend-cann-nnal_${version}_linux-aarch64.run |
  | Ascend-cann-nnal_${version}_linux-x86_64.run  |
  | ...                                          |


- 安装
    ```shell
    chmod +x Ascend-cann-nnal_*_linux-*.run
    ./Ascend-cann-nnal_*_linux-*.run -h # （可选）查看参数说明
    ./Ascend-cann-nnal_*_linux-*.run --install
    source /usr/local/Ascend/nnal/atb/set_env.sh
    ```

### 1.3 安装PytorchAdapter

先安装torch 再安装torch_npu

#### 1.3.1 安装torch

- 下载

  | 包名 （其中`${version}`为实际版本）                   |
  | -------------------------------------------- |
  | torch-${version}+cpu-cp310-cp310-linux_x86_64.whl |
  | torch-${version}-cp310-cp10-linux_aarch64.whl     |
  | ...                                          |

  - 根据所使用的环境中的python版本以及cpu类型，选择对应版本的torch安装包。

- 安装
  ```bash
  # 安装torch 2.1.0 的python 3.10 的arm版本为例
  pip install torch-2.1.0-cp310-cp310-linux_aarch64.whl
  ```

#### 1.3.2 安装torch_npu

安装方法：

| 包名 （其中`${version}`为实际版本）  |
| --------------------------- |
| pytorch_v${version}_py311.tar.gz |
| pytorch_v${version}_py310.tar.gz |
| ...                         |

- 安装选择与torch版本以及python版本一致的npu_torch版本

```bash
# 安装 torch_npu，以 torch 2.1.0，python 3.10 的版本为例
tar -zxvf pytorch_v2.1.0_py310.tar.gz
pip install torch*_aarch64.whl
```

### 1.4 安装开源软件依赖
| 模型                     | 开源软件依赖文件                                             |
| ------------------------ | ------------------------------------------------------------ |
| 默认依赖                 | [requirements.txt](./requirements/requirements.txt)           |
| Baichuan                 | [requirements_baichuan.txt](./requirements/models/requirements_baichuan.txt) |
| Bloom                    | [requirements_bloom.txt](./requirements/models/requirements_bloom.txt)      |
| Deepseek-Moe             | [requirements_deepseek_moe.txt](./requirements/models/requirements_deepseek_moe.txt) |
| Deepseek-vl              | [requirements_deepseek_vl.txt](./requirements/models/requirements_deepseek_vl.txt)      |
| Internvl                 | [requirements_internvl.txt](./requirements/models/requirements_internvl.txt) |
| Llama3                   | [requirements_llama3.txt](./requirements/models/requirements_llama3.txt) |
| Llava                    | [requirements_llava.txt](./requirements/models/requirements_llava.txt) |
| Qwen2_audio              | [requirements_qwen2_audio.txt](./requirements/models/requirements_qwen2_audio.txt) |
| Qwen2_vl                 | [requirements_qwen2_vl.txt](./requirements/models/requirements_qwen2_vl.txt) |
| Qwen2.5                  | [requirements_qwen2.5.txt](./requirements/models/requirements_qwen2.5.txt) |
| Qwen2                    | [requirements_qwen2.txt](./requirements/models/requirements_qwen2.txt) |
| Yi                       | [requirements_yi.txt](./requirements/models/requirements_yi.txt)           |



- 开源软件依赖请使用下述命令进行安装：
  ```bash
  pip install -r ./requirements/requirements.txt
  pip install -r ./requirements/models/requirements_{模型}.txt
  ```
- 各个模型开源软件依赖请使用上述表格各模型依赖的requirement_{模型}.txt进行安装，若模型未指定开源软件依赖，请使用默认开源软件依赖。

### 1.5 安装MindIE LLM
- 场景一：获取MindIE的整包进行安装
  - 下载编译好的包
    - [下载链接](https://www.hiascend.com/developer/download/community/result?module=ie+pt+cann)
    
    | 包名 （其中`${version}`为实际版本）       |
    | -------------------------------------- |
    | Ascend-mindie_${version}_linux-x86_64_abi{0}.run |
    | Ascend-mindie_${version}_linux-aarch64_abi{0}.run |
    | Ascend-mindie_${version}_linux-x86_64_abi{1}.run |
    | Ascend-mindie_${version}_linux-aarch64_abi{1}.run |

    - 可以使用`uname -m`指令查看服务器是x86还是aarch架构
    - 可以使用以下指令查看abi是0还是1
        ```shell
        python -c "import torch; print(torch.compiled_with_cxx11_abi())"
        ```
        - 若输出结果为True表示abi1，False表示abi0
  - 安装
    ```shell
    chmod +x Ascend-mindie*.run
    ./Ascend-mindie-{version}_linux-{arch}.run --install
    source /usr/local/Ascend/mindie/set_env.sh
    ```
- 场景二：手动编译MindIE LLM
  - 代码编译
  见MindIE_LLM开发指导
  - MindIE LLM安装
  ```shell
  cd output/${ARCH} # 其中ARCH为系统架构
  ./Ascend-mindie-llm_${VERSION}_linux-${ARCH}.run --install
  source /usr/local/Ascend/mindie_llm/set_env.sh
  ```

### 1.6 安装模型仓
- 场景一：使用编译好的包进行安装
  - 下载编译好的包
    - [下载链接](https://www.hiascend.com/developer/download/community/result?module=ie+pt+cann)

    | 包名  （其中`${version}`为实际版本）                           |
    | ------------------------------------------------------------ |
    | Ascend-mindie-atb-models_${version}_linux-aarch64_py310_torch2.1.0-abi0.tar.gz |
    | Ascend-mindie-atb-models_${version}_linux-aarch64_py310_torch2.1.0-abi1.tar.gz |
    | ...                                                          |

  - 将文件放置在/usr/local/Ascend/路径下
  - 解压
    ```shell
    mkdir /usr/local/Ascend/atb-models
    cd /usr/local/Ascend/atb-models
    tar -zxvf ../Ascend-mindie-atb-models_*_linux-*_torch*-abi*.tar.gz
    ```
- 场景二：手动编译模型仓
  - 获取模型仓代码
    ```shell
    git clone https://gitcode.com/ascend/MindIE-LLM.git
    ```
  - 切换到目标分支
    ```shell
    cd MindIE-LLM
    git checkout master
    ```
  - 代码编译
    ```shell
    cd examples/atb_models
    bash scripts/build.sh
    cd output/atb_models/
    source set_env.sh
    ```


### 1.6 安装量化工具msModelSlim (可选)
  - 量化权重需使用该工具生成，具体生成方式详见各模型README文件
  - 工具下载及安装方式见[README](https://gitcode.com/ascend/msit/blob/master/msmodelslim/README.md)

## 环境变量参考

### CANN、加速库、模型仓的环境变量
```shell
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
# 若使用编译好的包（即1.5章节的场景一），则执行以下指令
source /usr/local/Ascend/mindie/set_env.sh
source /usr/local/Ascend/atb-models/set_env.sh
```


### 多机推理
- 在性能测试时开启"AIV"提升性能，若有确定性计算需求时建议关闭"AIV"
```shell
   export HCCL_OP_EXPANSION_MODE="AIV"
```
- 若要在运行时中断进程，ctrl+C无效，需要使用pkill结束进程

### 显存分析
- 开启虚拟内存，提高内存碎片利用率
```shell
   export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
```

## 问题定位
- 若遇到推理执行报错，优先打开日志环境变量，并查看算子库和加速库的日志中是否有error级别的告警，基于error信息进一步定位
- 开启日志环境变量后，日志仍没有保存，定位排查点：
  - 1. 确认脚本中没有覆盖日志相关的环境变量
  - 2. 确认服务器内存充足
### 日志开关
- 模型仓日志
  - 打开模型仓日志
    - 推荐使用
      ```shell
      export MINDIE_LOG_TO_STDOUT=1
      export MINDIE_LOG_TO_FILE=1
      export MINDIE_LOG_LEVEL=info
      ```
    - 复杂场景
      ```shell
      export MINDIE_LOG_TO_STDOUT='llm:true;false'
      # 将llm组件打屏设置为true，其他组件设置为false
      export MINDIE_LOG_TO_FILE='llm:true;service:false'
      # 将llm组件落盘设置为true，service组件设置为false
      export MINDIE_LOG_LEVEL='llm:critical;service:debug'
      # 将llm组件日志界别设置为critical，service组件日志界别设置为debug
      export ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=1
      # 开启ATB流同步，以定位算子问题，多流场景下无法开启
      ```
  - 关闭模型仓日志
    - 推荐使用
      ```shell
      export MINDIE_LOG_TO_STDOUT=0
      export MINDIE_LOG_TO_FILE=0
      export ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=0
      ```
  - 日志存放在~/mindie/log/debug下

- CANN日志
  - 打开CANN日志
    ```shell
    export ASCEND_SLOG_PRINT_TO_STDOUT=1 #CANN日志是否输出到控制台 
    export ASCEND_GLOBAL_LOG_LEVEL=1 #CANN日志级别
    export ASCEND_MODULE_LOG_LEVEL=OP=1 #加速库日志级别
    # 0：对应DEBUG级别。
    # 1：对应INFO级别。
    # 2：对应WARNING级别。
    # 3：对应ERROR级别。
    # 4：对应NULL级别，不输出日志
    ```
  - 关闭加速库日志
    ```shell
    export ASCEND_SLOG_PRINT_TO_STDOUT=0
    export ASCEND_GLOBAL_LOG_LEVEL=3
    ```
  - 日志存放在~/ascend/log下
  - 加速库日志存放在~/ascend/log/atb下
  - plog日志存放在~/ascend/log/debug/plog下

- 注意：
    1、开启日志后会影响推理性能，建议默认关闭；当推理执行报错时，开启日志定位原因
    2、INFO级别日志和ASCEND_SLOG_PRINT_TO_STDOUT开关同时打开时，会让控制台输出大量打印，请根据需要打开。

### 精度问题定位

#### 确定性计算（可选）
- NPU支持确定性计算，即多次运行同样的数据集，结果一致
  - 确定性计算需设置以下环境变量
    ```shell
    export LCCL_DETERMINISTIC=1
    export HCCL_DETERMINISTIC=true
    export ATB_MATMUL_SHUFFLE_K_ENABLE=0
    export CLOSE_MATMUL_K_SHIFT=1
    ```
  - 关闭确定性计算
    ```shell
    export LCCL_DETERMINISTIC=0
    export HCCL_DETERMINISTIC=false
    export ATB_MATMUL_SHUFFLE_K_ENABLE=0
    ```
- 注意：开启确定性计算会影响性能；服务化场景下，动态batchsize无法保证精度是确定的。
- 若使用相同的Prompt构造不同Batch size进行推理时，为了保证结果一致，除了开启以上环境变量外，需关闭通信计算掩盖功能
  ```shell
  export ATB_LLM_LCOC_ENABLE=0
  ```

#### Dump Tensor
- 适用于定位精度问题
- msit推理工具提供dump tensor以及精度对比的能力
  - 工具下载方式见[README](https://gitcode.com/ascend/msit/blob/master/msit/docs/install/README.md)
  - 工具使用方式见[README](https://gitcode.com/ascend/msit/blob/master/msit/docs/llm/%E5%B7%A5%E5%85%B7-DUMP%E5%8A%A0%E9%80%9F%E5%BA%93%E6%95%B0%E6%8D%AE%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.md)

#### 溢出检查
- 可以开启以下环境变量，开启后若出现溢出，溢出值会被置为NaN；若不开启此变量，则会对溢出值进行截断
```shell
export INF_NAN_MODE_ENABLE=1
```
- 注意：Atlas 800I A2服务器若需要强制对溢出值进行截断（即`export INF_NAN_MODE_ENABLE=0`），需要额外将`INF_NAN_MODE_FORCE_DISABLE`环境变量设置为1。

### 性能调优
#### Profiling
- 适用于性能分析（例如：查看单token耗时前三的算子）及定位性能问题（例如：算子间有空泡，单算子耗时过长）
- Step 1：开启Profiling环境变量
  ```shell
  export ATB_PROFILING_ENABLE=1
  ```
  - Profiler Level默认使用Level0
- Step 2：执行推理
  - 运行推理指令（此指令和不生成Profiling文件时运行的指令无差异）
  - 注意：生成Profiling文件时，增量token数量不应设置太大，否则会导致Profiling文件过大
- Step 3：下载单卡Profiling文件
  - Profiling文件默认保存在${cur_dir}/profiling目录下
    - 可以通过设置`PROFILING_FILEPATH`环境变量进行修改
    - 多卡运行时${cur_dir}/profiling目录下会有多个文件夹
  - 单卡Profiling文件在${cur_dir}/profiling/\*/PROF\*/mindstudio_profiler_output路径下，名称为msprof_\*.json
- Step 4：查看Profiling文件
  - 将msprof_\*.json文件拖拽入[Mind-Studio-Insight](https://www.hiascend.com/developer/download/community/result?module=pt+sto+cann)中即可查看Profiling文件
- ${cur_dir}/profiling/\*/PROF\*/mindstudio_profiler_output路径下名为op_summary_\*.csv的文件中有单算子更加详细的性能数据
  - 开启`export ATB_PROFILING_TENSOR_ENABLE=1`环境变量后可以在此文件中查看算子执行时的shape信息（注意：开启此环境变量会导致CPU耗时增加，影响整体流水，但对单算子分析影响不大）
  - 获取shape信息需将Profiler Level设置为Level1，可以通过以下环境变量设置
    ```shell
    export PROFILING_LEVEL=Level1
    ```
  - 如需获取算子调用栈信息，请修改`{$atb_models_path}/examples/run_pa.py`中的`with_stack`字段为True

#### 开启透明大页
- 若Profiling出现快慢进程，导致通信算子耗时长、波动大；模型整网性能测试，同一组测试样例，测试多次性能波动较大；此时可以尝试开启透明大页，提升内存访问性能。
- 开启方式
  ```shell
  echo always > /sys/kernel/mm/transparent_hugepage/enabled
  ```
- 关闭方式
  ```shell
  echo never > /sys/kernel/mm/transparent_hugepage/enabled
  ```
- 备注：
  - 仅ARM服务器适用
  - 指令请在裸机上执行


#### 开启CPU Performance模式
- 开启CPU Performance模式以提高模型推理性能（首次开启时，根据提示安装依赖）
  ```
  cpupower -c all frequency-set -g performance
  ```
- 备注：
  - 仅ARM服务器适用
  - 指令请在裸机上执行

## 公网地址说明
- 代码涉及公网地址参考[此README文档](./public_address_statement.md)

## 约束
- 使用ATB Models进行推理，模型初始化失败时，请结束进程。
- 使用ATB Models进行推理，权重路径及文件的权限需保证其他用户无写权限
