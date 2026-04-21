# FAQ

## 常见的LLM推理性能优化手段都有哪些

算子融合、量化、Tensor并行、ContinuousBatching等。

## 纯模型推理时出现“out of memory, need block”报错

### 问题现象描述

纯模型推理时，报错出现“out of memory, need block”，具体报错信息示例如下图：

![](./figures/faq_out_of_memory.png)

### 原因分析

通常是由于大图片或者视频导致的序列增长，导致预分配的KV cache不够用。

### 解决措施

在“run\_pa.sh”脚本中修改“max\_input\_length”，根据实际应用场景，设置一个更大的值。

## 单机Atlas 800I A3 超节点服务器进行PD混部服务部署时，出现chat接口性能劣化

### 问题现象描述

单卡专家数较多，平均每个专家分到token不多，当精度有差异的场景下，有可能遇到chat接口比非chat接口性能更差的情况。

### 原因分析

chat接口激活的专家分布更均衡，但单卡激活的专家数更多，需要搬运的专家也多，会导致性能变差，造成GMM算子性能波动。

### 解决措施

这个是chat/no chat接口导致的固有差异，属于正常现象。

## 当出现undefinedsymbols: xxx这样的报错如何定位

## 解决方案

需要确认MindIE LLM和ATB，CANN，torch，torch\_npu是否匹配，ABI=0/1的选择是否正确 。

## 多卡服务化分布式推理时缺失环境变量MASTER\_ADDR或MASTER\_PORT

### 问题现象描述

当模型在模型侧使用torch.distributed多卡分布时，从服务侧拉起出现没有MASTER\_ADDR或者MASTER\_PORT环境变量：

![](./figures/faq_master_addr.png)

### 原因分析

没有设置环境变量MASTER\_ADDR或MASTER\_PORT。

### 解决措施

可以通过如下两种方式设置环境变量：

- 在代码中设置：

 ```python
    import os

    os.environ\['MASTER\_ADDR'\] = 'localhost'

    os.environ\['MASTER\_PORT'\] = '5678'

```

- 通过环境变量设置：

  ```bash
    export MASTER\_ADDR=localhost

    export MASTER\_PORT=5678

  ```

## 服务侧拉起模型时出现“Max retries exceeded with url”报错

### 问题现象描述

服务侧拉起模型时出现“Max retries exceeded with url”报错，具体报错信息如下：

![](./figures/faq_max_retries_exceeded.png)

### 原因分析

大概率是内网访问的问题。

### 解决措施

以Qwen-VL为例，打开权重文件夹下tokenization\_qwen.py文件，按照如下29\~30行修改：

![](./figures/faq_max_retries_exceeded2.png)

## 服务化加载模型后出现“Socket bind failed”报错

### 问题现象描述

Service侧加载模型后快速退出程序，出现“Socket bind failed”报错，具体日志如下：

![](./figures/faq_socked_bind_failed.png)

### 原因分析

MindIE Motor使用HTTP或者HTTPS协议进行通信，让客户端先断开连接可以减少服务器的负担，保证资源的合理释放，包括端口资源等。

### 解决措施

进入配置文件修改对应的“port”、“managementPort”和“metricsPort”参数。为避免该类问题发生，请在断开服务进程之前确保先断开请求。或使用**lsof -i :_\{port\}_**查看端口状态，若仍被残留进程占用，使用kill信号清理残留进程。其中_\{port\}_替换为查看的端口号。

## 服务化拉起后发送请求无响应

### 问题现象描述

服务侧成功拉起，发送请求却迟迟无法得到响应。

### 原因分析

可查看模型侧日志，可能是out of memory。

### 解决措施

在配置文件中，更改“npuDeviceIds”和“npuMemSize”。

## 服务化拉起失败，如何查看日志定位问题

若服务化拉起失败，优先查看日志信息，日志路径默认在“/root/mindie/log/debug”下。

## 服务化拉起LLaMA2-13b-hf失败，显示core dump，报错显示和protobuf相关，怎么解决

可尝试protobuf升级，pip install protobuf==5.28。

## 拉起服务化时出现“Check\_path: config.json failed”报错

### 问题现象描述

拉起服务化时，遇到Check\_path: config.json failed报错，具体报错信息如下：

![](./figures/faq_check_path_configjson_failed.png)

### 原因分析

模型权重路径下的模型配置文件“config.json”没有640的权限。

### 解决措施

可通过如下两种方法修改权限：

- 执行如下命令，修改“config.json文件”的权限，：

    ```python
    chmod 640 {model_path}/config.json
    ```

- 执行如下命令，修改模型权重整个文件夹的权限：

    ```python
    chmod -R 640 {model_path}
    ```

## 拉起服务时，出现“pybind11::error\_already\_set”报错

### 问题现象描述

拉起服务时，出现“pybind11::error\_already\_set”报错，具体报错信息如下：

![](./figures/faq_pybind11_error.png)

### 原因分析

模型侧的三方依赖不正确，此时需要重新安装模型依赖的三方包。

## 解决措施

根据模型的“requirements.txt”文件，重新安装第三方依赖，依赖文件默认路径为：“{MindIE安装目录}/atb_llm/requirements/models/requirements\__\{model\}._txt”

## 拉起服务时core dump无报错日志

### 问题现象描述

拉起服务时core dump无报错日志。

### 原因分析

模型侧的三方依赖不正确，如Protobuf等，此时需要重新安装模型依赖的三方包。

### 解决措施

根据模型的“requirements.txt”文件，重新安装第三方依赖，依赖文件默认路径为：“{MindIE安装目录}/atb_llm/requirements/models/requirements\__\{model\}_.txt”

## 如何开启加速库的强制同步来定位报错

### 解决方案

```bash
export ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=1
export ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE=1
export ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE=1
```

开启环境变量后，运行模型推理，根据加速库日志中的第一个error进一步定位。

## 什么是确定性计算

确定性计算，是指在输入数据集等输入条件不变时，多次运行推理应用，输出结果每次保持一致。

## 为什么跑精度数据集的时候，最后的精度结果有浮动

1. 将模型后处理部分从sampling改为greedy后，可以基本保证输出文本的稳定性。
2. 由于确定性计算的问题，输出可能存在略微差异。

## 为什么相同输入，组batch顺序不同，送入LLM模型推理输出不同

### 解决方案

1. 由于matmul算子在不同行上的累加顺序不完全相同，加之浮点精度没有加法交换律的特性，导致不同行上即使输入完全相同，计算结果也会存在一定的误差。
2. 可以通过设置环境变量export ATB\_MATMUL\_SHUFFLE\_K\_ENABLE=0将加速库matmul的shuffle k功能关闭，关闭之后可以保证所有行上算子累加顺序一致，但matmul性能会下降10%左右 。

## 为什么相同输入送入MindIE Server推理，输出存在一定的不确定性

调度框架代码（block查询模式下）是可以保证确定性的，但是在CPU负载等环境因素的影响下，可能会对请求到达时间产生影响，最终影响调度的确定性。例如，客户外部服务查询引擎后得知可以提交10个请求，第一次运行时，10个请求快速到达并组成batch；但第二次运行时，受环境影响部分请求到达较晚，仅有5个请求组成了batch；那么两次运行的结果就会产生差异。

## 异步执行出现定位困难的报错

### 问题现象描述

纯模型推理时遇到定位困难的报错。

### 原因分析

由于模型推理中有异步执行，导致报错信息可能不是真实报错信息，需要同步后进行定位。

### 解决措施

设置环境变量“export ASCEND\_LAUNCH\_BLOCKING=1”，打开同步运行后再进行定位。

## 在昇腾上进行LLM推理，如何保证确定性计算

确定性计算，是指在输入数据集等输入条件不变时，多次运行推理应用，输出结果每次保持一致。

1. 模型层面：

    通信算子：

    ```bash
    export LCCL_DETERMINISTIC=1
    export HCCL_DETERMINISTIC=true
    ```

    MatMul：

    ```bash
    export ATB_MATMUL_SHUFFLE_K_ENABLE=0
    ```

2. 推理引擎：

    MindIE：基于block进行新request的获取。

    TGI：暂不支持。

## LLM推理结果存在乱码

### 解决方案

检查tokenizer在token转id时，是否使用了正确的模型路径。

## PD分离场景，D节点出现“Pull kv failed”报错日志

### 问题现象描述

推理过程中，D节点在拉取KV cache时，出现“Pull kv failed”的“ERROR”级别报错日志，并且CANN的status\_code中出现了timeout的错误码。

![](./figures/ScreenShot_20250427162525.png)

### 原因分析

PD分离场景中，D节点的KV cache需要从P节点那里拉取，出现这个错误，说明从P到D的KV cache传输超时，极有可能是网络质量差导致的。

### 解决措施

![](./figures/ScreenShot_20250427162356.png)

- （推荐）使用如下命令查看网络重传次数，如果有部分卡网络重传次数过高，请检查该光模块。

    ```python
      for i in $(seq 0 7); do echo "============> $i";hccn_tool -i $i -stat -g |grep rty;done
    ```

- 在MindIE的配置文件“ModelDeployConfig”字段中设置"kv\_trans\_timeout" 为“5”，表示Pull kv的超时时间为5秒。这样设置可能会掩盖由网络问题导致的推理性能问题，请谨慎设置。

## 部署MindIE LLM服务时出现LLMPythonModel initializes fail的报错提示

### 问题描述

部署MindIE LLM服务时，出现LLMPythonModel initializes fail的报错，如下图所示。

![](./figures/faq001.png)

### 原因分析

ibis缺少Python依赖。

### 解决步骤

进入/_Service安装路径_/logs目录，打开Python日志，根据日志报错信息，安装需要的依赖。

## 加载模型时出现out of memory报错提示

*### 问题描述

部署MindIE LLM服务，加载模型时出现out of memory报错提示，如下图所示。

![](./figures/faq002.png)

### 原因分析

权重太大，内存不足。

### 解决步骤

将config.json文件中ModelConfig的npuMemSize调小，比如调成8。

## 部署MindIE LLM服务时，出现atb\_llm.runner无法import报错

### 问题描述

部署MindIE LLM服务时，出现atb\_llm.runner无法import，如下图所示。

![](./figures/faq003.png)

### 原因分析

由于Python版本不是配套版本3.10，或者pip对应的Python版本不是目标版本3.10，找不到对应的包。可以通过python和pip -v查看对应的Python版本进行确认。

### 解决步骤

1. 使用以下命令打开bashrc文件。

    ```bash
    vim ~/.bashrc
    ```

2. 在bashrc文件内添加如下环境变量，保存并退出。

    ```bash
    ## 例如系统使用3.11版本，安装目录位于/usr/local/python3.11
    export LD_LIBRARY_PATH=/usr/local/python3.11/lib:$LD_LIBRARY_PATH
    export PATH=/usr/local/python3.11/bin:$PATH
    ```

3. 使用以下命令使环境变量生效。

    ```bash
    source ~/.bashrc
    ```

4. 使用以下命令建立软链接。

    ```bash
    ln -s /usr/local/python3.11/bin/python3.11 /usr/bin/python
    ln -s /usr/local/python3.11/bin/pip3.11 /usr/bin/pip
    ```

## 部署MindIE LLM服务时，找不到tlsCert等路径

### 问题描述

部署MindIE LLM服务时，找不到tlsCert等路径，如下图所示。

![](./figures/faq004.png)

### 原因分析

开启HTTPS服务时，未将需要的证书放到对应的目录下。

### 解决步骤

将生成服务器证书、CA证书、和服务器私钥等认证需要的文件，放置在对应的目录。

## 使用第三方库transformers跑模型推理时，报错“cannot allocate memory in static TLS block”

### 问题描述

报错详细信息如下所示：

![](./figures/faq005.png)

### 原因分析

glibc.so本身的bug。

### 解决步骤

执行以下命令：

```bash
export LD_PRELOAD=$LD_PRELOAD:/usr/local/python3.11/lib/python3.11/site-packages/torch/lib/../../torch.libs/libgomp-6e1a1d1b.so.1.0.0

```

## 加载大模型时耗时过长<a id="jzdmxshsgc"></a>

### 问题描述

加载1300B大小的模型时耗时过长（约3个小时）。其中"B"代表"Billion"，即十亿。

### 原因分析

未使用异步加载。

### 解决方法

通过设置环境变量OMP\_NUM\_THREADS进行模型加载优化，OMP\_NUM\_THREADS用于设置OpenMP（Open Multi-Processing）并行编程框架的线程数量，设置后加载1300B大小的模型只要10分钟左右。

```bash
export OMP_NUM_THREADS=1
```

此外，使用下面命令启动NPU显存碎片收集。

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.96
```

## 多模态模型输入时，出现大小限制报错问题

### 问题描述

多模态模型输入（image\_url/video\_url/audio\_url）时出现类似以下报错提示：

- OpenAI接口：

    ```text
    {"error": "Message len not in (0, 4194304], but the length of inputs is xxxxx", "error_type": "Input Validation Error"}
    ```

- vLLM接口：

    ```text
    Prompt must be necessary and data type must be string and length in (0, 4194304], but the length of inputs is xxxxx
    ```

- Triton接口：

    ```text
    Text_input must be necessary and data type must be string and length in (0, 4194304], but the length of inputs is xxxxx
    ```

<br>

### 原因分析

可能输入的图片、音频或者视频是base64编码格式（Base64 编码后的数据通常是原始数据的4/3倍），导致整个message/prompt/text\_input超过4MB而出现报错。

<br>

### 解决方案

- 方式一：参考接口约束

    - OpenAI接口：

        请求参数中的messages参数下所有字段的字符数被限制为不大于4MB，详情请参见**推理接口**。

    - vLLM接口：

        请求参数中的prompt参数下所有字段的字符数被限制为不大于4MB，详情请参见《MindIE LLM开发指南》中的“API接口说明 \> RESTful API参考 \> EndPoint业务面RESTful接口 \> 兼容vLLM 0.6.4版本接口 \> 文本/流式推理接口”。

    - Triton接口：

        请求参数中的text\_input参数下所有字段的字符数被限制为不大于4MB，详情请参见《MindIE LLM开发指南》中的“API接口说明 \> RESTful API参考 \> EndPoint业务面RESTful接口 \> 兼容Triton接口 \> 文本推理接口”。

    >[!NOTE]说明
    >- 假如当base64编码的图片格式大小为1MB，message/prompt/text\_input下其余的请求字符数大于3MB时，也会导致整个message/prompt/text\_input超过4MB而出现报错。
    >- 如果image\_url/video\_url/audio\_url传的是本地图片/视频/音频，或者是一个远端的url，这个url的字符串长度大小加上message/prompt/text\_input下其余的请求字符数大小也应该小于4MB。url传进来之后，会去加载解析该url得到该图片/视频/音频：
    >    - 图片：不大于20MB；
    >    - 视频：不大于512MB；
    >    - 音频：不大于20MB；
    >- base64编码的输入更容易超出限制，当前版本会报错， 因为涉及到安全问题，建议采用网页地址或本地地址。
    >- 如果选择使用base64格式请求，请勿直接使用终端curl，建议选择使用Python脚本，因为base64编码后的数据字符长度可能超出系统终端限制，导致请求被截断。

- 方式二：手动修改源码

    比如修改输入inputs大小的限制为10MB，代码修改示例如下所示：

    **图 1**  示例一

    ![](./figures/faq006.png)

    **图 2**  示例二

    ![](./figures/faq007.png)

## 多模态模型推理服务时报错：RuntimeError: call calnnCat failed, detail:EZ1001

### 问题描述

多模态模型推理服务时，文件MindIE-LLM-master\\examples\\atb\_models\\atb\_llm\\models\\qwen2\_vl\\flash\_causal\_qwen2\_using\_mrope.py出现类似以下报错提示：

```text
call calnnCat failed, detail:EZ1001: xxxxxxxx dimnum of tensor 5 is [1], should be equal to tensor 0 [2].
```

**图 1**  报错信息

![](./figures/faq008.png)

**图 2**  报错文件

![](./figures/faq009.png)

**图 3**  报错文件

![](./figures/faq010.png)

<br>

### 原因分析

可能是concat相关的某个算子，tensor 5在某个维度上是1，与要求的维度上是2大小不一致。可能与squeeze有关，因为squeeze会去掉大小为1的维度。

示例：如果某个算子（如：concat、matmul等），希望这个维度存在并匹配某个值（如：2），那被squeeze删除后shape就会报错。

>[!NOTE]说明
>MindIE 2.0之前的版本存该问题，MindIE 2.0版本之后都已修复。

<br>

### 解决方案

修改代码，如下所示：

![](./figures/faq013.png)

## 运行Qwen2.5-VL系列模型失败并报错

### 问题描述

运行Qwen2.5-VL系列模型失败并出现类似以下报错提示：

- 报错提示一：

    ```text
    You are using a model of type qwen2_5_vl to instantiate a model of type. This is not supported for all configurations of models and can yiled errors.
    ```

- 报错提示二：

    ```text
    [ERROR] TBE Subprocess[task_distribute] raise error[], main process disappeared!
    ```

<br>

### 原因分析

模型配置等不支持，一般是因为安装的依赖不正确，需要安装对应的依赖文件。

<br>

### 解决方案

- 报错提示一处理方式：

    根据每个模型所需依赖安装对应的requirements.txt 文件。

    - 所有模型需要安装的通用依赖文件所在路径为：

        ```text
        {MindIE安装目录}/atb_llm/requirements/requirements.txt
        ```

    - 不同的模型对应的依赖安装文件在models路径下，比如qwen2-vl 模型：

        ```text
        {MindIE安装目录}/atb_llm/requirements/models/requirements_qwen2_vl.txt
        ```

    安装命令如下所示：

    ```bash
    pip install -r {MindIE安装目录}/atb_llm/requirements/models/requirements_qwen2_vl.txt
    ```

- 报错提示二处理方式：

    1. 单击[链接](https://modelers.cn/models/MindIE/DeepSeek-R1-Distill-Llama-70B)查看该模型硬件环境是否支持。
    2. 使用以下命令排查驱动版本是否配套，驱动最低版本23.0.7才能运行，建议安装驱动版本为24.1.RC2及以上。

        ```bash
        npu-smi info
        ```

    3. 初始环境变量检查下是否都已经配置好，并且已经生效。
    4. 检查系统空闲内存是否充足。

        使用以下命令查看free的内存大小，需保证大于**权重大小/机器数**。

        ```bash
        free -h
        ```

        根据经验，尽量保证**free\_mem \>= \(权重大小/机器数\) \* 1.3**。

        >[!NOTE]说明
        >每次跑完模型，请检查一下机器的host侧内存占用，避免内存不足导致模型运行失败。

    5. 导入以下环境变量：

        ```bash
        export HCCL_DETERMINISTIC=false
        export HCCL_OP_EXPANSION_MODE="AIV"
        export NPU_MEMORY_FRACTION=0.96
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        ```

    6. 排查多机服务化参数配置是否一致。
    7. 重启服务器，并重新启动服务。

    >[!NOTE]说明
    >- 硬件环境、版本配套，驱动、镜像等版本更新到最新版能有效避免很多类似此类报错问题。
    >- 该报错更多处理方式请参见[链接](https://www.hiascend.com/developer/blog/details/02112175404775067102)。

## 纯模型推理和服务化拉起或推理时，出现Out of memory（OOM）报错

### 问题描述

纯模型推理和服务化拉起/推理时，出现各种Out of memory（OOM）报错，报错信息类似如下所示：

```text
RuntimeError: NPU out of memory. Tried to allocate xxx GiB."
```

<br>

### 原因分析

- 模型权重文件较大。
- 用户输入shape超大（batch size较大、过长的文本、过大的图片、音频或视频）。
- 配置文件参数配置超大。

<br>

### 解决方案

1. 适当调高NPU\_MEMORY\_FRACTION环境变量的值（代表内存分配比例，默认值为0.8），参考示例如下所示。

    ```bash
    export NPU_MEMORY_FRACTION=0.96
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_NUM_THREADS=1
    ```

2. 适当调低服务化配置文件config.json中maxSeqLen、maxInputTokenLen、maxPrefillBatchSize、maxPrefillTokens、maxBatchSize等参数的值，主要是调整maxPrefillTokens、maxSeqLen和maxPrefillTokens参数。
    - maxPrefillTokens需要大于等于maxInputToken。
    - maxPrefillTokens会影响到atb初始化阶段的workspace，其值过大时拉起服务后可能直接出现Out of memory报错。

3. 调整npuMemSize（代表单个NPU可以用来申请KV Cache的size上限，默认值为-1，表示自动分配KV Cache；大于0表示手动分配，会根据设置的值固定分配KV Cache大小）参数值。

    npuMemSize = 单卡总内存 \* 内存分配比例

4. 使用更多显卡，比如当前使用2张卡，可以适当增加至4张或者8张，前提是需要确认当前模型在当前硬件中支持几张卡。

## 多模态模型推理时报错：Qwen2VL/Qwen2.5VL_VIT_graph nodes[1] infershape fail

### 问题描述

多模态模型进行推理时出现类似以下报错提示：

```text
[standard_model.py:188] : [Model] >>> global rank-2 Execute type:1, Exception:Qwen25VL_VIT_graph nodes[1] infershape  fail, enable log: export ASDOPS_LOG_LEVEL=ERROR, export ASDOPS_LOG_TO_STDOUT=1
```

或者：

```text
[error] [1256320] [operation_base.cpp:273] Qwen25VL_VIT_layer_0_graph infer shape fail, error code: 8
```

<br>

### 原因分析

- 使用的模型在当前版本可能不支持该硬件环境。

- 输入shape过大，self-attention算子不支持。

<br>

### 解决方案

- 使用的模型在当前版本可能不支持该硬件环境
    - 单击[链接](https://www.hiascend.com/software/mindie/modellist)，查看MindIE各版本模型支持度，选择正确的MindIE版本。
    - 通过修改代码临时解决，可直接在镜像中修改相应的代码，只需要修改python代码，不需要重新编译，如下所示。

        ![](./figures/faq011.png)

- 输入shape过大，self-attention算子不支持

    将服务化配置文件config.json中的maxPrefillTokens参数适当调小。

## 多模态模型进行推理时，出现输入image_url/video_url/audio_url格式报错问题

### 问题描述

多模态模型输入image\_url/video\_url/audio\_url格式进行推理时，出现类似以下报错提示：

```text
File "/usr/local/lib/python3.11/site-packages/atb_llm/examples/models/qwen2_vl/run_pa.py", line 365, in <module>    raise TypeError("The multimodal input field currently only supports 'image' and 'video'.")
```

<br>

### 原因分析

image\_url/video\_url/audio\_url参数中的值未符合指定的要求。

<br>

### 解决方案

**Image**

1. 格式一：\{"type": "image\_url", "image\_url": image\_url\}， 此类格式的image\_url支持本地路径、jpg图片的base64编码、http和https协议url。

2. 格式二：\{"type": "image\_url", "image\_url": \{"url": \{image\_url\}\}\}，此类格式的image\_url支持本地路径、jpg图片的base64编码、http和https协议url。

3. 格式三：\{"type": "image\_url", "image\_url": \{"url": "file://\{local\_path\}"\}，此类格式仅支持本地路径。

4. 格式四：\{"type": "image\_url", "image\_url": \{"url": f"data:<mime\_type\>/<subtype\>;base64,<base64\_data\>"\}\}，此类格式仅支持base64编码，源格式可以为jpg、jpeg、png，对应的MIME如下表所示。

|格式|MIME|
|--|--|
|jpg|image/jpeg|
|jpeg|image/jpeg|
|png|image/png|

**Video**

1. 格式一：\{"type": "video\_url", "video\_url": video\_url\}， 此类格式的video\_url支持本地路径、http和https协议url。

2. 格式二：\{"type": "video\_url", "video\_url": \{"url": \{video\_url\}\}\}，此类格式的video\_url支持本地路径、http和https协议url。

3. 格式三：\{"type": "video\_url", "video\_url": \{"url": "file://\{local\_path\}"\}，此类格式仅支持本地路径。

4. 格式四：\{"type": "video\_url", "video\_url": \{"url": f"data:<mime\_type\>/<subtype\>;base64,<base64\_data\>"\}\}，此类格式仅支持base64编码，源格式可以为mp4、avi、wmv，对应的MIME如下表所示。另，由于视频编码的后的长度可能超出MindIE Service服务化请求字符长度的最大上限，因此并不建议使用base64编码格式传输视频。

|格式|MIME|
|--|--|
|mp4|video/mp4|
|avi|video/x-msvideo|
|wmv|video/x-ms-wmv|

**Audio**

1. 格式一：\{"type": "audio\_url", "audio\_url": audio\_url\}， 此类格式的audio\_url支持本地路径、http和https协议url。
2. 格式二：\{"type": "audio\_url", "audio\_url": \{"url": \{audio\_url\}\}\}，此类格式的audio\_url支持本地路径、http和https协议url。
3. 格式三：\{"type": "audio\_url", "audio\_url": \{"url": "file://\{local\_path\}"\}}，此类格式仅支持本地路径。
4. 格式四：\{"type": "audio\_url", "audio\_url": \{"url": f"data:<mime\_type\>/<subtype\>;base64,<base64\_data\>"\}\}，此类格式仅支持base64编码，源格式可以为mp3、wav、flac，对应的MIME如下表所示。

    |格式|MIME|
    |--|--|
    |mp3|audio/mpeg|
    |wav|audio/x-wav|
    |flac|audio/flac|

5. 格式五：\{"type": "input\_audio", "input\_audio": \{"data": f"\{audio\_base64\}", "format": "wav"\}\}，当type为input\_audio时，仅支持base64编码格式，源格式支持mp3、wav、flac，同时，必须通过format字段明确源格式类型。

## 运行Qwen2-VL系列模型时报错：Failed to get vocab size from tokenizer wrapper with exception

### 问题描述

Qwen2-VL系列模型Tokenizer报错（其他模型也有可能报错，报错无关于模型），出现类似以下报错提示：

```text
Failed to get vocab size from tokenizer wrapper with exception...
```

**图 1**  报错提示

![](./figures/faq012.png)

<br>

### 原因分析

- 模型适配的transformers/tokenizer版本不正确。
- trust\_remote\_code参数配置为false。
- 服务化的config.json文件、模型权重路径和模型config.json文件权限不正确。
- 模型权重文件下可能缺少config.json文件。
- 词表文件损坏。

<br>

### 解决方案

- 模型适配的transformers/tokenizer版本不正确。
    - 确认每个模型需要安装依赖的transformers的版本，一般在模型文件的requirements.txt文件中可查看。然后查看模型权重路径下的config.json文件中的transformers版本号是否一致。
    - 使用以下tokenizer校验方法，创建一个Python脚本，如果运行成功，则tokenizer加载无问题。

        ```python
        from transformers import AutoTokenizer  tokenizer = AutoTokenizer.from_pretrained('path/to/model')
        ```

- trust\_remote\_code参数配置为false。
    - 将trust\_remote\_code参数配置为true。

- 服务化的config.json文件、模型权重路径和模型config.json文件权限不正确。
  - 将服务化的config.json文件、模型权重路径和模型config.json文件权限修改为640。

- 模型权重文件下可能缺少config.json文件。
    - 如果缺少config.json文件，将其补齐。

- 词表文件损坏。
    - 使用以下命令检查tokenizer.json文件的完整性

        ```bash
        sha256sum tokenizer.json # 哈希校验，输出的值和原权重文件进行对比
        ```

## MindIE部署Qwen2.5系列模型执行量化推理时报错

### 问题描述

MindIE部署Qwen2.5系列模型执行量化推理时出现以下报错信息：

```text
ValueError: linear type not matched,please check 'config.json' 'quantize' parameter
```

或

```text
AttributeError: 'ForkAwareLocal' object has no attribute 'connection'
```

<br>

### 原因分析

未配置quantize字段。

<br>

### 解决方案

执行量化推理时，需要在量化权重所在路径的config.json文件中添加quantize字段，值为当前量化权重的量化方式，示例如下：

```text
"quantize": "w8a8"
```

## 使用MindIE进行推理时，如何保证每次推理结果的一致性

### 问题描述

使用MindIE进行推理，相同输入，组batch顺序不同时，模型推理输出结果不同。

<br>

### 原因分析

由于matmul算子在不同行上的累加顺序不完全相同，且浮点精度没有加法交换律的特性，导致不同行上即使输入完全相同，计算结果也会存在一定的误差。

<br>

### 解决方案

确定性计算，是指在输入数据集等输入条件不变时，多次运行推理应用，输出结果每次保持一致。可以通过设置环境变量export ATB\_MATMUL\_SHUFFLE\_K\_ENABLE=0将加速库matmul的shuffle k功能关闭，关闭之后可以保证所有行上算子累加顺序一致，但matmul性能会下降10%左右 。

通信算子：

```bash
export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=true #开启归约类通信算子的确定性计算
export ATB_LLM_LCOC_ENABLE=0
```

MatMul：

```bash
export ATB_MATMUL_SHUFFLE_K_ENABLE=0
```

## Gloo连接失败报错：ERROR failed to connect. error=SO_ERROR: Connection refused

### 问题描述

启动MindIE LLM服务时，出现Gloo连接错误，日志显示类似信息：

```text
ERROR failed to connect, willRetry=1, retry=2, retryLimit=3, rank=1, size=2, local=[127.0.0.1]:123, remote=[127.0.0.1]:345, error=SO_ERROR: Connection refused
```

### 原因分析

该错误通常出现在**多机部署场景**下，核心原因是Gloo组件自动选择了错误的网络接口（网卡），导致节点间通信失败。

### 解决方案

通过环境变量显式指定Gloo使用的网络接口：

1. **查看可用网卡**：
   在每个节点执行以下命令查看网卡名称：

   ```bash
   # Linux系统
   ip addr
   # 或
   ifconfig
   ```

   常见网卡命名格式：`enp*`、`ens*`、`eth*`等

2. **设置环境变量**：
   在启动服务前，为每个节点设置`GLOO_SOCKET_IFNAME`环境变量：

   ```bash
   export GLOO_SOCKET_IFNAME=<网卡名称>  # 例如：export GLOO_SOCKET_IFNAME=enp1s0
   ```

3. **容器部署注意事项**：
   - 使用Docker时，需将容器网络模式设置为`host`模式
   - 每个机器的网卡名称可能不同，需分别配置为本机的网卡名称

4. **Kubernetes部署注意事项**：
   - Kubernetes集群中，网卡名称通常映射为`eth0`
   - 大EP部署时，可以在`boot_helper/boot.sh`脚本中配置环境变量

### 验证

设置完成后重新启动服务，检查是否仍出现Gloo连接错误。
