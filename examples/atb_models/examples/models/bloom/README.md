# BLOOM

* [BLOOM](https://huggingface.co/bigscience/bloom) (BigScience Large Open-science Open-access Multilingual Language Model)
* 此代码仓中实现了一套基于 NPU 硬件的 BLOOM 推理模型。

## 特性矩阵

- 此矩阵罗列了各 BLOOM 模型支持的特性：

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化 | KV cache量化 | 稀疏量化 | MOE量化 | MindIE Service | TGI |  长序列 |
|-------------|----------------------------|-----------------------------|------|----------------------|-----------------|-----------------|---------|-----------|--------------|--------------------------|-----|--------|-----|-----|
| bloom-7b1 | 支持world size 1,2,4,8   | 支持world size 1,2,4 | 是   | 否  | 否              | 是              | 否       | 否       | 否           | 否       | 否     | 否    | 否  | 否  |

## 推理使用说明

### 路径变量解释

| 变量名        | 含义                                                         |
| ------------- | ------------------------------------------------------------ |
| `working_dir` | 加速库及模型库下载后放置的目录                               |
| `llm_path`    | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/MindIE-LLM/`；若使用gitcode下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models` |
| `script_path` | 脚本所在路径。BLOOM 系列模型的工作脚本所在路径为`{llm_path}/examples/models/bloom` |
| `weight_path` | HF 原始模型权重路径（`.safetensors` 格式）                   |

权重下载链接：

* bloom-7b1: https://huggingface.co/bigscience/bloom-7b1/tree/main

> 下载权重时无需下载 `pytorch_model.bin.index.json` 以及 `.bin` 文件。

框架加载权重时会从下载的 `config.json` 里面读取 `torch_dtype`，因此需要手动在 `config.json` 里面补上 `"torch_dtype": "float16"`。

### 环境准备

1、安装 CANN 8.0 的环境，并 `source /path/to/cann/set_env.sh`；

2、使用 Python 3.9 或更高；

3、使用 torch 2.0 或更高版本，并安装对应的 torch_npu；

4、安装依赖：

```shell
pip install transformers==4.34.0
pip install accelerate
```

5、安装 `atb_llm`:

```shell
cd $llm_path
python setup.py bdist_wheel
python -m pip install dist/*.whl --force-reinstall
```

## BLOOM-7B1

### 权重准备

在 Hugging Face 上下载模型权重文件（推荐下载 `.safetensors`，`.bin` 需要转换成 `.safetensors`），权重路径为 `weight_path`。

### PagedAttention模型

进入 `modeltest` 路径下：

```shell
cd tests/modeltest
```

进行测试前需要先设置一些环境变量：

```shell
export HCCL_BUFFSIZE=110
export PYTHONWARNINGS="ignore"
export ATB_OPERATION_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1
export ATB_USE_TILING_COPY_STREAM=1
export ATB_CONTEXT_WORKSPACE_RING=1
export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
```

#### 性能测试

> `$weight_path` 可以是 HuggingFace 原始权重路径，也可以是量化后的模型权重路径（下同）。

```shell
bash run.sh pa_fp16 performance [[seq_in,seq_out],[seq_in,seq_out]] $batch_size bloom $weight_path $tp
```

例如：`TP = 8`，`batch_size = 1`：

```shell
bash run.sh pa_fp16 performance [[256,256],[512,512],[1024,1024],[2048,2048]] 1 bloom /path/to/model 8
```

#### 下游任务精度测试

```shell
bash run.sh pa_fp16 full_CEval $n_shot $batch_size bloom $weight_path $tp
```

例如：`TP = 8`，`batch_size = 1`，`CEval 5-shot`：

```shell
bash run.sh pa_fp16 full_CEval 5 1 bloom /path/to/model 1
```

更详细的配置选项请参考：`examples/atb_models/tests/modeltest/README.md`
