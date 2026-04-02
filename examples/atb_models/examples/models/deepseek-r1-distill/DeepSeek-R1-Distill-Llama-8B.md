# DeepSeek-R1-Distill-Llama-8B
DeepSeek-R1-Distill-Llama-8B 为Deepseek利用由 DeepSeek-R1 生成的推理数据，对稠密模型Llama3进行了微调。评估结果显示，提炼后的小型稠密模型在基准测试中的表现非常出色。

# 特性矩阵

- 下表展示Llama模型各版本支持的特性

| 模型及参数量      | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化 | KV cache量化 | 稀疏量化（仅支持300I DUO） | MOE量化 | MindIE Service | TGI | 长序列 | prefix_cache | FA3量化 | functioncall | Multi LoRA|
| ----------------- |----------------------------|-----------------------------| ---- | ---- | --------------- | --------------- | -------- | --------- | ------------ | -------- | ------- | -------------- | --- | ------ | ---------- | --- | --- | --- |
| DeepSeek-R1-Distill-Llama-8B           | 支持world size 1,2,4,8       | 支持world size 2,4,8              | √    | √    | ×               | √               | √        | ×         | ×            | √        | ×       | √              | ×   | ×      | x       | x | x | x |

注：表中所示支持的world size为对话测试可跑通的配置，实际运行时还需考虑输入序列长度带来的显存占用。

## 权重

**权重下载**

- [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/tree/main)


## 路径变量解释

| 变量名  | 含义                                             |
|--------|--------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                  |
| llm_path | ATB_Models模型仓所在路径；若使用编译好的包，则路径为`${working_dir}/`；若使用gitcode下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models/`    |
| script_path | 脚本所在路径；Llama3系列的工作脚本所在路径为`${llm_path}/examples/models/llama3`                            |
| weight_path | 模型权重路径                            |

## 量化权重生成
### Atlas 800I A2 w8a8量化
* 生成量化权重依赖msModelSlim工具，安装方式见[此README](https://gitcode.com/ascend/msit/tree/master/msmodelslim)

* 量化权重统一使用${ATB_SPEED_HOME_PATH}/examples/convert/model_slim/quantifier.py脚本生成，以下提供Llama模型量化权重生成快速启动命令

* W8A8量化权重请使用以下指令生成
    * 注意该量化方式仅支持在Atlas 800I A2服务器上运行

```shell
# 设置CANN包的环境变量
source /usr/local/Ascend/cann/set_env.sh
# 关闭虚拟内存
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
# 进入atb-models目录
cd ${ATB_SPEED_HOME_PATH}
# DeepSeek-R1-Distill-Llama-8B量化，有回退层，antioutlier使用m1算法配置，使用min-max量化方式，校准数据集使用50条BoolQ数据，在NPU上进行运算
bash examples/models/llama3/generate_quant_weight.sh -src {浮点权重路径} -dst {W8A8量化权重路径} -type llama3.1_8b_w8a8
```

### Atlas 300I DUO稀疏量化
**Step 1 生成W8A8S量化权重**
- 注意该量化方式仅支持在Atlas 300I DUO卡上运行
- 修改模型权重config.json中`torch_dtype`字段为`float16`
- 生成量化权重依赖msModelSlim工具，安装方式见[此README](https://gitcode.com/ascend/msit/tree/master/msmodelslim)
- 进入到{msModelSlim工具路径}/msit/msmodelslim/example/Llama的目录 `cd msit/msmodelslim/example/Llama`；
```shell
# 运行量化转换脚本
python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8S量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 4 --a_bit 8 --fraction 0.011 --co_sparse True
```

**Step 2 量化权重切分及压缩**
- 该步骤需要在Atlas 300I DUO卡上运行
```shell
# 执行"jq --version"查看是否安装jq，若返回"bash：jq：command not found"，则依次执行"apt-get update"和"apt install jq"
jq --version
```
```shell
export IGNORE_INFER_ERROR=1
# 进入atb-models目录
cd ${ATB_SPEED_HOME_PATH}
# 运行切分及压缩脚本
torchrun --nproc_per_node {TP数} -m examples.convert.model_slim.sparse_compressor --model_path {W8A8S量化权重路径} --save_directory {W8A8SC量化权重路径}
```
- TP数为tensor parallel并行个数
- 注意：若权重生成时以TP=2进行切分，则运行时也需以TP=2运行
- 示例
```shell
torchrun --nproc_per_node 2 -m examples.convert.model_slim.sparse_compressor --model_path /data1/weights/model_slim/Llama-8b_w8a8s --save_directory /data1/weights/model_slim/Llama-8b_w8a8sc
```

## 纯模型推理

### 对话测试
进入llm_path路径

```shell
cd $ATB_SPEED_HOME_PATH
```

执行对话测试

```shell
torchrun --nproc_per_node 2 \
         --master_port 20037 \
         -m examples.run_pa \
         --model_path ${权重路径} \
         --input_texts 'What is deep learning?' \
         --max_output_length 20
```

### 性能测试
进入ModelTest路径
```shell
cd $ATB_SPEED_HOME_PATH/tests/modeltest/
```
运行测试脚本
```shell
bash run.sh pa_[data_type] performance [case_pair] [batch_size] ([prefill_batch_size]) [model_name] ([is_chat_model]) (lora [lora_data_path]) [weight_dir] ([trust_remote_code]) [chip_num] ([parallel_params]) ([max_position_embedding/max_sequence_length])
```
具体执行batch=1, 输入长度256, 输出长度256用例的4卡并行性能测试命令如下，

Atlas 800I A2:
```shell
bash run.sh pa_bf16 performance [[256,256]] 1 llama ${weight_path} 4
```

Atlas 300I Duo: 
```shell
bash run.sh pa_fp16 performance [[256,256]] 1 llama ${weight_path} 4
```

> 注：ModelTest为大模型的性能和精度提供测试功能。使用文档请参考`${ATB_SPEED_HOME_PATH}/tests/modeltest/README.md`
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
"port" : 1025, #自定义
"managementPort" : 1026, #自定义
"metricsPort" : 1027, #自定义
...
"httpsEnabled" : false,
...
},

"BackendConfig": {
...
"npuDeviceIds" : [[0,1,2,3]],
...
"ModelDeployConfig":
{
"ModelConfig" : [
{
...
"modelName" : "llama",
"modelWeightPath" : "/data/datasets/DeepSeek-R1-Distill-Llama-8B",
"worldSize" : 4,
...
}
]
},
...
}
}
```

- 拉起服务化

```shell
cd /usr/local/Ascend/mindie/latest/mindie-service/bin
./mindieservice_daemon
```

- 新建窗口测试(VLLM接口)

```shell
curl 127.0.0.1:1025/generate -d '{
"prompt": "What is deep learning?",
"max_tokens": 32,
"stream": false,
"do_sample":true,
"temperature": 0.6,
"top_p": 0.95,
"model": "llama"
}'
```

> 注: 服务化推理的更多信息请参考[MindIE Service用户指南](https://www.hiascend.com/document/detail/zh/mindie/100/mindieservice/servicedev/mindie_service0001.html)

## 常见问题
1. ImportError: cannot import name 'shard_checkpoint' from 'transformers.modeling_utils'. 降低transformers版本可解决。

```shell
pip install transformers==4.46.3
pip install numpy==1.26.4
```