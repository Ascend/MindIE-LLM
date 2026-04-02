# 离线推理

## ATB Models纯模型使用

### 前提条件

已在环境上安装CANN、PyTorch、Torch-NPU和ATB Models，详情请参见《MindIE安装指南》。

> [!NOTE]说明 
> 本次样例参考以下安装路径进行：
> 安装ATB Models并初始化ATB Models环境变量。模型仓set\_env.sh脚本中有初始化“\$\{ATB\_SPEED\_HOME\_PATH\}”环境变量的操作，所以source模型仓中set\_env.sh脚本时会同时初始化“\$\{ATB\_SPEED\_HOME\_PATH\}”环境变量。

### 约束

-  使用ATB Models进行推理，模型初始化失败时，模型初始化过程中用户自定义修改导致的失败，需要手动结束进程。
-  使用ATB Models进行推理，权重路径及文件的权限需保证其他用户无写权限。

### README文档解读

当前ATB Models包含三类Readme文档指导您执行推理流程，了解模型支持特性以及提供基础的调测和问题定位手段。

**图 1** ATB Models  Readme文档关系示意图<a name="fig89051134504"></a>  
![](./figures/atb_models_readme.png "ATB-Models-Readme文档关系示意图")

**表 1**  Readme文档介绍

|文档名称|作用|内容|
|--|--|--|
|${ATB_SPEED_HOME_PATH}/README.md|为ATB Models所有文档的总入口。|<ul><li>运行ATB Models依赖的硬件和软件版本。每个模型所依赖的软件版本不同，请根据对应的\$\{ATB_MODELS_HOME_PATH\}/requirements进行安装，详细信息见\$\{ATB_SPEED_HOME_PATH\}/README.md文档。</li><li>基本调测和问题定位手段：<br>算子库、加速库和模型仓日志开启方式；<br>性能分析方法；<br>精度分析方法。</li><li>预置模型列表：<br>此处会链接至模型的README文档。</li></ul>|
|${ATB_SPEED_HOME_PATH}/examples/models/{模型名称}/README.md|为ATB Models每个模型各自的文档，<br>例如：${ATB_SPEED_HOME_PATH}/examples/models/llama/README.md中为LLaMA模型的文档，其中涵盖了LLaMA系列和LLaMA2系列模型的介绍和运行指导。|<ul><li>模型特性支持矩阵，即不同参数量的模型对各类硬件，各种量化方式，各种特性的支持情况。</li><li>模型开源权重下载地址。</li><li>模型量化权重生成介绍。</li><li>对话测试、精度测试和性能测试脚本执行方式。</li></ul>|
|${ATB_SPEED_HOME_PATH}/examples/README.md|汇总了对于公共能力和接口的介绍。|<ul><li>bin格式的权重转safetensor格式脚本的介绍。</li><li>量化权重生成脚本的介绍。</li><li>Flash Attention和Paged Attention启动脚本参数介绍。</li><li>可选择性配置的环境变量介绍。</li><li>特殊场景注意事项。</li></ul>|


### 使用示例

下面以LLaMA3-8B模型为例，展示对话推理以及性能测试的执行步骤。

1.  配置环境变量。

    ```bash
    # 配置CANN环境，默认安装在/usr/local目录下
    source /usr/local/Ascend/cann/set_env.sh
    # 配置加速库环境
    source /usr/local/Ascend/nnal/atb/set_env.sh
    # 配置模型仓环境变量
    source /usr/local/Ascend/atb-models/set_env.sh
    ```

2.  准备模型权重：可从Hugging Face官网直接下载，将下载的权重保存在“/data/Llama-3-8b“。
3.  执行如下命令，修改权重文件权限。

    ```bash
    chmod -R 755 /data/Llama-3-8b
    ```

4.  （可选）当前ATB Models推理仅支持加载safetensor格式的权重文件。若下载的权重文件是safetensor格式文件，则无需进行权重转换，若下载的权重文件是bin格式文件，则需要按照如下方式进行转换。

    ```bash
    # 进入ATB Models 所在路径
    cd ${ATB_SPEED_HOME_PATH}
    # 执行脚本生成safetensor格式的权重
    python examples/convert/convert_weights.py --model_path /data/Llama-3-8b
    ```

    输出结果会保存在bin格式的权重文件同目录下。

5.  测试对话推理。

    ```bash
    cd ${ATB_SPEED_HOME_PATH}
    bash examples/models/llama/run_pa.sh /data/Llama-3-8b
    ```

    如上命令调用的run\_pa.sh脚本是对run\_pa.py脚本的封装，默认推理内容为"What's deep learning?"，batch size为1，可以通过下方的步骤[6](#step6)修改推理内容。

6.  <a name="step6"></a>自定义推理内容。

    -  用户可以通过以下方式直接调用run\_pa.py脚本，通过传入参数的方式自定义推理内容及推理方式。

        例如：使用/data/Llama-3-8b路径下的权重，使用8卡推理"What's deep learning?"和"Hello World."，推理时batch size为2。

        ```bash
        # 指定当前机器上可用的逻辑NPU核心，多个核心间使用逗号相连
        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        # 执行推理
        torchrun --nproc_per_node 8 --master_port 20030 -m examples.run_pa --model_path /data/Llama-3-8b --input_texts "What's deep learning?" "Hello World." --max_batch_size 2
        ```

        > [!NOTE]说明
        > 环境变量说明请参见[环境变量说明](environment_variable.md)。

    -  用户可以通过传入Token id的方式进行推理。

        新建一个py脚本（如test.py）用于生成Token id：

        ```python
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path="{tokenizer所在的文件夹路径}",
            use_fast=False,
            padding_side='left',
            trust_remote_code="{用户输入的trust_remote_code值}")
        inputs = tokenizer("What's deep learning?", return_tensors="pt")
        token_id = inputs.data["input_ids"]
        print(token_id)
        ```

        执行如下命令，生成Token id：

        ```python
        python test.py
        ```

        执行如下命令进行推理，如下以生成的第一个推理内容对应的Token id为"1,15043,2787"，第二个推理内容对应的Token id为"1,306,626,2691"为例，其中推理内容间以空格分开。

        ```linux
        # 执行推理
        torchrun --nproc_per_node 8 --master_port 20030 -m examples.run_pa --model_path /data/Llama-3-8b --input_ids 1,15043,2787 1,306,626,2691 --max_batch_size 2
        ```

        **表 2**  run\_pa.py脚本参数说明  <a id="table2"></a>

        |参数名称|是否为必选|类型|默认值|描述|
        |--|--|--|--|--|
        |--model_path|是|string|""|模型权重路径。该路径会进行安全校验，必须使用绝对路径，且和执行推理用户的属组和权限保持一致。|
        |--input_texts|否|string|"What's deep learning?"|推理文本或推理文本路径，多条推理文本间使用空格分割。|
        |--input_ids|否|string|None|推理文本经过模型分词器处理后得到的token id列表，多条推理请求间使用空格分割，单个推理请求内每个token使用逗号隔开。|
        |--input_file|否|string|None|仅支持jsonl格式文件，每一行必须为List[Dict]格式的按时间顺序排序的对话数据，每个Dict字典中需要至少包含"role"和"content"两个字段。|
        |--input_dict|否|parse_list_of_json|None|推理文本以及对应的adapter名称。格式形如：'[{"prompt": "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?", "adapter": "adapter1"}, {"prompt": "What is deep learning?", "adapter": "base"}]'|
        |--max_prefill_batch_size|否|int或者None|None|模型推理最大Prefill Batch Size。|
        |--max_position_embeddings|否|int或者None|None|模型可接受的最大上下文长度。当此值为None时，则从模型权重文件中读取。|
        |--max_input_length|否|int|1024|推理文本最大token数。|
        |--max_output_length|否|int|20|推理结果最大token数。|
        |--max_prefill_tokens|否|int|-1|模型Prefill推理阶段最大可接受的token数。若输入为-1，则max_prefill_tokens = max_batch_size * (max_input_length + max_output_length)|
        |--max_batch_size|否|int|1|模型推理最大batch size。|
        |--block_size|否|int|128|KV Cache分块存储，每块存储的最大token数，默认为128。|
        |--chat_template|否|string或者None|None|对话模型的prompt模板。|
        |--ignore_eos|否|bool|store_true|当推理结果中遇到eos token（句子结束标识符）时，是否结束推理。若传入此参数，则忽略eos token。|
        |--is_chat_model|否|bool|store_true|是否支持对话模式。若传入此参数，则进入对话模式。|
        |--is_embedding_model|否|bool|store_true|是否为embedding类模型。默认为因果推断类模型，若传入此参数，则为embedding类模型。|
        |--load_tokenizer|否|bool|True|是否加载tokenizer。若传入False，则必须传入input_ids参数，且推理输出为token id。|
        |--enable_atb_torch|否|bool|store_true|是否使用Python组图。默认使用C++组图，若传入此参数，则使用Python组图。|
        |--kw_args|否|string|""|扩展参数，支持用户通过扩展参数进行功能扩展。|
        |--trust_remote_code|否|bool|store_true|是否信任模型权重路径下的自定义代码文件。默认不执行。若传入此参数，则transformers会执行用户权重路径下的自定义代码文件，这些代码文件的功能的安全性需由用户保证，请提前做好安全性检查。|
        |--dp|否|int|-1|数据并行数，默认不进行数据并行。|
        |--tp|否|int|-1|整网张量并行数，若值为“-1”，默认张量并行数为worldSize值。|
        |--sp|否|int|-1|序列并行数，默认不进行序列并行。若开启序列并行数，一般与张量并行数保持一致。|
        |--cp|否|int|-1|文本并行数，默认不进行文本并行。|
        |--moe_tp|否|int|-1|稀疏模型MoE模块中的张量并行数，默认等于“tp”数。若同时配置“tp”参数，则“moe_tp”参数优先级高于“tp”参数。|
        |--moe_ep|否|int|-1|稀疏模型MoE模块中的专家并行数，默认无专家并行。|
        |--lora_modules|否|string|None|定义需要加载的Lora权重名以及对应的Lora权重路径。例如：'{"adapter1": "/path/to/lora1", "adapter2": "/path/to/lora2"}'。默认不加载Lora权重。|
        |--max_loras|否|int|0|LoRA场景中，定义最多可存储的LoRA数量。动态LoRA场景下必须配置，静态LoRA场景中可以不配置。若传入数值过大，由于预留了过多权重空间，会出现out_of_memory报错信息，例如: "RuntimeError: NPU out of memory. Tried to allocate xxx GiB."|
        |--max_lora_rank|否|int|0|动态加载卸载LoRA场景中，定义最大LoRA秩。动态LoRA场景下必须配置，静态LoRA场景中可以不配置。若传入数值过大，由于预留了过多权重空间，会出现out_of_memory报错信息，例如: "RuntimeError: NPU out of memory. Tried to allocate xxx GiB."|


        > [!NOTE]说明

        > 此章节中的run\_pa.py脚本用于纯模型快速测试，脚本中未增加强校验，出现异常情况时，会直接抛出异常信息。例如：
        > - nput\_texts、input\_ids、input\_file、input\_dict参数包含推理内容，程序进行数据处理的时间和传入数据量成正比。同时这些输入会被转换成token id搬运至NPU，传入数据量过大可能会导致这些NPU tensor占用显存过大，而出现由out of memory导致的报错信息，例如："req: xx input length: xx is too long, max\_prefill\_tokens: xx"等报错信息。
        > - chat\_template参数可以使用两种形式输入：模板文本或模板文件的路径。当以模板文本输入时，若文本长度过大，可能会导致运行缓慢。
        > - 脚本会基于max\_batch\_size、max\_input\_length、max\_output\_length、max\_prefill\_batch\_size和max\_prefill\_tokens等参数申请推理输入及KV Cache，若用户传入数值过大，会出现由out of memory导致的报错信息，例如："RuntimeError: NPU out of memory. Tried to allocate xxx GiB."。
        > - 脚本会基于max\_position\_embeddings参数，申请旋转位置编码和attention mask等NPU tensor，若用户传入数值过大，会出现由out of memory导致的报错信息，例如："RuntimeError: NPU out of memory. Tried to allocate xxx GiB."。
        > - block\_size参数若小于张量并行场景下每张卡实际分到的注意力头个数，会出现由shape不匹配导致的报错（"Setup fail, enable log: export ASDOPS\_LOG\_LEVEL=ERROR, export ASDOPS\_LOG\_TO\_STDOUT=1 to find the first error. For more details, see the MindIE official document."），需开启日志查看详细信息。

7.  测试性能。

    开启ATB\_LLM\_BENCHMARK\_ENABLE环境变量后，将统计模型首Token、增量Token及端到端推理时延。

    ```bash
    # 环境变量开启方式
    export ATB_LLM_BENCHMARK_ENABLE=1
    # 启动推理方式见步骤4、步骤5
    ```

    耗时结果会显示在Console中，并保存在./benchmark\_result/benchmark.csv文件里。

    > [!NOTE]说明 
    > 性能测试后，可使用msprof工具，进行性能数据采集和性能数据分析，达到性能调优目的。msprof工具的使用可参见《性能调优工具》的“[msprof命令行工具](https://www.hiascend.com/document/detail/zh/mindstudio/700/T&ITools/Profiling/atlasprofiling_16_0010.html)”章节。

 
## ATB Models服务化使用 

### 前提条件

已在环境上安装CANN、PyTorch、Torch-NPU、ATB Models、MindIE LLM和MindIE Motor，详情请参见《MindIE安装指南》。

## 使用实例

1.  设置环境变量。

    若安装路径为默认路径，可以运行以下命令初始化各组件环境变量。

    ```bash
    # 配置CANN环境，默认安装在/usr/local目录下
    source /usr/local/Ascend/cann/set_env.sh
    # 配置加速库环境
    source /usr/local/Ascend/nnal/atb/set_env.sh
    # 配置模型仓环境变量
    source /usr/local/Ascend/atb-models/set_env.sh
    # MindIE
    source /usr/local/Ascend/mindie/latest/mindie-llm/set_env.sh
    source /usr/local/Ascend/mindie/latest/mindie-service/set_env.sh
    ```

2.  启动服务化并发送请求。

    MindIE服务化使用方法请参考《MindIE Motor开发指南》中的“快速入门 \> [启动服务](https://gitcode.com/Ascend/MindIE-Motor/blob/dev/docs/zh/user_guide/quick_start.md)”章节。服务化参数配置请参考[配置参数说明（服务化）](service_parameter_configuration.md)。

    服务化配置中默认使用ATB Models作为模型后端。

    ```bash
    vim /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
    # ModelDeployConfig.ModelConfig.backendType字段默认值为"atb"
    "backendType": "atb"
    ```

    服务化API接口请参考《MindIE Motor开发指南》中的“服务化接口”章节。

    用户可使用HTTPS客户端（Linux curl命令，Postman工具等）发送HTTPS请求，此处以Linux curl命令为例进行说明。重开一个窗口，使用以下命令发送请求。

    ```bash
    curl -H "Accept: application/json" -H "Content-type: application/json" -X POST --cacert {Server服务端证书的验签证书/根证书路径} --cert {客户端证书文件路径} --key {客户端证书私钥路径} -d '{"inputs": "hi","stream":false}' https://{ip}:{port}/generate
    ```
