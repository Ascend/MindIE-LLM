
# DeepSeek-R1-Distill-Qwen-7B

DeepSeek-R1-Distill-Qwen-7B 为 DeepSeek 利用由 DeepSeek-R1 生成的推理数据，对密集型模型 Qwen2.5 进行了微调。评估结果显示，提炼后的小型密集模型在基准测试中的表现非常出色。

## 特性矩阵

- 下表展示 Qwen 模型各版本支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8 量化 | W8A16 量化 | KV cache 量化 | 稀疏量化（仅支持 300I DUO） | MOE 量化 | MindIE Service | TGI | 长序列 | prefix_cache | FA3 量化 | functioncall | Multi LoRA |
| ----------------- |----------------------------|-----------------------------| ---- | ---- | --------------- | --------------- | -------- | --------- | ------------ | -------- | ------- | -------------- | --- | ------ | ---------- | --- | --- | --- |
| DeepSeek-R1-Distill-Qwen-7B | 支持 world size 1,2,4,8 | 支持 world size 2,4,8 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

注：表中所示支持的 world size 为对话测试可跑通的配置，实际运行时还需考虑输入序列长度带来的显存占用。

- qwen2/2.5 系列模型在 800I A2 仅支持 bfloat16 浮点类型; 300I DUO 仅支持 float16 浮点类型, 需要修改权重目录下的 `config.json` 文件，**"torch_dtype"字段改为"float16"**
- 稀疏量化 w8a8sc 仅支持在 300I DUO 上使用
- 稀疏量化分为两个步骤。步骤一：w8a8s 可在任何机器上生成，注意 config 中需要将"torch_dtype"改为"float16"。800I A2 机器上可以使用多卡进行量化生成 w8a8s 权重。300I DUO 上仅支持单卡或 cpu 生成 w8a8s 权重。步骤二：w8a8sc 需要在 300I DUO 上切分。

## 权重

**权重下载**

- [DeepSeek-R1-Distill-Qwen-7B](https://modelers.cn/models/State_Cloud/DeepSeek-R1-Distill-Qwen-7B)

## 路径变量解释

| 变量名      | 含义                                                                                       |
|------------|-------------------------------------------------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录（仅源码下载场景使用）                                         |
| llm_path    | 模型仓所在路径。若使用镜像，则路径为 `/usr/local/Ascend/atb-models`；若使用 gitcode 下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径。QWen 系列模型的工作脚本所在路径为 `${llm_path}/examples/models/qwen`           |
| weight_path | 模型权重路径                                                                                 |

## 权重量化

### Atlas 800I A2 w8a8 量化

W8A8 量化权重可通过 [msmodelslim Qwen](https://gitcode.com/Ascend/msmodelslim/tree/26.1.0/example/Qwen)（昇腾模型压缩工具）实现。

- 注意该量化方式仅支持在 Atlas 800I A2 服务器上运行
- 请参考 [msmodelslim](https://gitcode.com/Ascend/msmodelslim/blob/26.1.0/docs/zh/install_guide/install_guide.md) 安装 msModelSlim 量化工具
- 进入到 msmodelslim/example/Qwen 的目录 `cd msmodelslim/example/Qwen`；并在进入的 Qwen 目录下，运行量化转换脚本

```bash
python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8 量化权重路径} --calib_file ../common/teacher_qualification.jsonl --w_bit 8 --a_bit 8 --device_type npu --anti_method m4
```

- 请将{浮点权重路径} 和{量化权重路径} 替换为用户实际路径。
- 如果需要使用 npu 多卡量化，请先配置环境变量，支持多卡量化,建议双卡执行量化：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
```

### Atlas 300I DUO/Atlas 300I Pro/Atlas 300V 稀疏量化

- Step 1
  - 注意该量化方式仅支持在 Atlas 300I DUO/Atlas 300I Pro/Atlas 300V 卡上运行
  - 修改模型权重 config.json 中 `torch_dtype` 字段为 `float16`
  - 请参考 [msmodelslim](https://gitcode.com/Ascend/msmodelslim/blob/26.1.0/docs/zh/install_guide/install_guide.md) 安装 msModelSlim 量化工具
  - 进入到 msmodelslim/example/Qwen 的目录 `cd msmodelslim/example/Qwen`；并在进入的 Qwen 目录下，运行量化转换脚本

> 注： 安装完 CANN 后 需要执行 source ${HOME}/Ascend/cann/set_env.sh 声明 ASCEND_HOME_PATH 值 后续安装 msmodelslim 前需保证其不为空
> 安装 CANN 时，如果用户未指定安装路径，则软件会安装到默认路径下，默认安装路径如下：root 用户：“/usr/local/Ascend”，非 root 用户：“${HOME}/Ascend”，${HOME} 为当前用户目录。

**Atlas 300I DUO**使用以下方式生成 W8A8S 量化权重

  ```shell
      export ASCEND_RT_VISIBLE_DEVICES=0
      export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
      python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8S 量化权重路径} --calib_file ../common/cn_en.jsonl --w_bit 4 --a_bit 8 --fraction 0.011 --co_sparse True --device_type npu --use_sigma True --is_lowbit True --sigma_factor 4.0 --anti_method m4
  ```

**Atlas 300I Pro/Atlas 300V**使用以下方式生成 W8A8S 量化权重

  ```bash
      python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8S 量化权重路径} --calib_file ../common/cn_en.jsonl --w_bit 4 --a_bit 8 --fraction 0.011 --co_sparse True --device_type cpu --use_sigma True --is_lowbit True --sigma_factor 4.0 --anti_method m4
  ```

> Atlas 300I Pro/Atlas 300V 量化过程耗时较长，预计 5 小时左右，可以在 Atlas 300I DUO 上先生成 W8A8S 量化权重路径，再搬运到 Atlas 300I Pro/Atlas 300V 执行后续步骤。

- Step 2：量化权重切分及压缩

  ```shell
    # 执行"jq --version"查看是否安装 jq，若返回"bash：jq：command not found"，则依次执行"apt-get update"和"apt install jq"
    jq --version
    export IGNORE_INFER_ERROR=1
    cd ${llm_path}
    torchrun --nproc_per_node {TP 数} -m examples.convert.model_slim.sparse_compressor --multiprocess_num 4 --model_path {W8A8S 量化权重路径} --save_directory {W8A8SC 量化权重路径}
  ```

  - TP 数为 tensor parallel 并行个数
  - 注意：若权重生成时以 TP=4 进行切分，则运行时也需以 TP=4 运行
  - 示例

  ```shell
  torchrun --nproc_per_node 4 -m examples.convert.model_slim.sparse_compressor --model_path /data1/weights/model_slim/Qwen-7b_w8a8s --save_directory /data1/weights/model_slim/Qwen-7b_w8a8sc
  ```

## 服务化推理

- 打开配置文件

```shell
vim /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
```

- 更改配置文件

```json
{
...
"ServerConfig" :
{
...
"port" : 1040, # 自定义
"managementPort" : 1041, # 自定义
"metricsPort" : 1042, # 自定义
...
"httpsEnabled" : false,
...
},

"BackendConfig": {
...
"npuDeviceIds" : [[0,1]],
...
"ModelDeployConfig":
{
"truncation" : false,
"ModelConfig" : [
{
...
"modelName" : "qwen",
"modelWeightPath" : "/data/datasets/DeepSeek-R1-Distill-Qwen-7B",
"worldSize" : 2,
...
}
]
},
}
}
```

- 拉起服务化

```shell
cd /usr/local/Ascend/mindie/latest/mindie-service/bin
./mindieservice_daemon
```

### AISBench 精度测试

精度测试使用 [AISBench](https://github.com/AISBench/benchmark) 工具。

**1、数据集准备**

请参考[数据集准备指南](https://github.com/AISBench/benchmark/blob/master/docs/source_zh_cn/get_started/datasets.md)获取开源数据集，下载解压后放置于 AISBench 工具根路径的 `datasets` 文件夹下。

以 MMLU 数据集为例：

```shell
cd /usr/local/lib/python3.11/site-packages/ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mmlu.zip
unzip mmlu.zip
rm mmlu.zip
```

**2、修改配置**

以 `<aisbench_install_path>/benchmark/configs/models/vllm_api/vllm_api_general_chat.py` 为例，配置内容示例如下：

```python
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="/path/to/DeepSeek-R1-Distill-Qwen-7B", # 模型权重文件夹路径
        model="qwen", # 与服务化 config.json 中的 modelName 一致
        request_rate=0,
        retry=2,
        host_ip="127.0.0.1", # 推理服务 IP
        host_port=1040, # 服务化 config.json 中配置的 port
        max_out_len=4096, # 最大输出 token 数
        batch_size=32, # 最大并发数
        trust_remote_code=False,
    )
]
```

**3、执行测试**

```shell
ais_bench --models vllm_api_general_chat --datasets gsm8k_gen_4_shot_cot_chat_prompt --debug
```

**4、常用数据集测试命令**

| 数据集 | max_out_len | 命令 |
|--------|-------------|------|
| gsm8k | 4096 | `ais_bench --models vllm_api_general_chat --datasets gsm8k_gen_4_shot_cot_chat_prompt --debug` |
| mmlu | 20 | `ais_bench --models vllm_api_general_chat --datasets mmlu_gen_5_shot_str --mode all` |
| ceval | 32000 | `ais_bench --models vllm_api_general_chat --datasets ceval_gen_0_shot_cot_chat_prompt --mode all` |

> 注意：MMLU 数据集中有约为 1.4w 条数据，推理耗时会比较长。`max_out_len` 设为 20 即可（选择题只需输出选项字母）。

### AISBench 性能测试

使用 [AISBench](https://github.com/AISBench/benchmark) 工具进行性能测试。

**1、配置 `vllm_api_stream_chat.py`**

```python
from ais_bench.benchmark.models import VLLMCustomAPIChatStream

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="", # 指定模型词表文件路径（可选，传空字符串会自动获取）
        model="qwen", # 与服务化 config.json 中的 modelName 一致
        request_rate=0, # 请求发送频率，<0.1 则一次性发送所有请求
        retry=2,
        host_ip="127.0.0.1", # 推理服务 IP
        host_port=1040, # 服务化 config.json 中配置的 port
        max_out_len=1024, # 最大输出 token 数
        batch_size=16, # 最大并发数
        generation_kwargs=dict(
            temperature=0,
            ignore_eos=True, # 测试定长输出时需设为 True
        )
    )
]
```

**2、执行性能测试**

```shell
ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_0_shot_cot_str_perf --mode perf --summarizer default_perf --debug
```

## 常见问题

1. ImportError: cannot import name 'shard_checkpoint' from 'transformers.modeling_utils'. 降低 transformers 版本可解决。

```shell
pip install transformers==4.46.3 --force-reinstall
pip install numpy==1.26.4 --force-reinstall
```bash

1. ImportError: cannot import name 'shard_checkpoint' from 'transformers.modeling_utils'. 降低 transformers 版本可解决。

```shell
pip install transformers==4.46.3 --force-reinstall
pip install numpy==1.26.4 --force-reinstall
```
