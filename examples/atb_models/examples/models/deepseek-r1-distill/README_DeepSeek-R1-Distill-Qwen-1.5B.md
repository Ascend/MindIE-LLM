
# DeepSeek-R1-Distill-Qwen-1.5B
DeepSeek-R1-Distill-Qwen-1.5B 为Deepseek利用由 DeepSeek-R1 生成的推理数据，对密集型模型Qwen2.5进行了微调。评估结果显示，提炼后的小型密集模型在基准测试中的表现非常出色。

# 特性矩阵

- 下表展示Qwen模型各版本支持的特性

| 模型及参数量      | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化 | KV cache量化 | 稀疏量化（仅支持300I DUO） | MOE量化 | MindIE Service | TGI | 长序列 | prefix_cache | FA3量化 | functioncall | Multi LoRA|
| ----------------- |----------------------------|-----------------------------| ---- | ---- | --------------- | --------------- | -------- | --------- | ------------ | -------- | ------- | -------------- | --- | ------ | ---------- | --- | --- | --- |
| DeepSeek-R1-Distill-Qwen-1.5B           | 支持world size 1,2,4,8       | 支持world size 2,4,8              | √    | √    | √               | √               | √        | ×         | ×            | ×        | ×       | √              | ×   | ×      | x       | x | x | x |

注：表中所示支持的world size为对话测试可跑通的配置，实际运行时还需考虑输入序列长度带来的显存占用。
- qwen2/2.5系列模型在800I A2仅支持bfloat16浮点类型; 300I DUO仅支持float16浮点类型, 需要修改权重目录下的`config.json`文件，**"torch_dtype"字段改为"float16"**
- 稀疏量化w8a8sc仅支持在300I DUO上使用
- 稀疏量化分为两个步骤。步骤一：w8a8s 可在任何机器上生成，注意config中需要将"torch_dtype"改为"float16"。800I A2机器上可以使用多卡进行量化生成w8a8s权重。300I DUO上仅支持单卡或cpu生成w8a8s权重。步骤二：w8a8sc 需要在300I DUO上切分。
## 权重

**权重下载**

- [DeepSeek-R1-Distill-Qwen-1.5B](https://modelers.cn/models/State_Cloud/DeepSeek-R1-Distill-Qwen-1.5B)


## 路径变量解释

| 变量名称    | 含义                                                                                                                                                   |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| working_dir | 加速库及模型库下载后放置的目录                                                                                                                         |
| llm_path    | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/MindIE-LLM/`；若使用gitcode下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径。QWen系列模型的工作脚本所在路径为`${llm_path}/examples/models/qwen`                                                                       |
| weight_path | 模型权重路径                                                                                                                                           |


## 权重量化
### Atlas 800I A2 w8a8量化
W8A8量化权重可通过[msmodelslim Qwen](https://gitcode.com/Ascend/msit/blob/master/msmodelslim/example/Qwen/README.md)（昇腾模型压缩工具）实现。
- 注意该量化方式仅支持在Atlas 800I A2服务器上运行
- 请参考[msmodelslim](https://gitcode.com/Ascend/msit/blob/master/msmodelslim/docs/%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97.md)安装msModelSlim量化工具
- 进入到msit/msmodelslim/example/Qwen的目录 `cd msit/msmodelslim/example/Qwen`；并在进入的Qwen目录下，运行量化转换脚本
```bash
python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/teacher_qualification.jsonl --w_bit 8 --a_bit 8 --device_type npu  --anti_method m4
```
- 请将{浮点权重路径}和{量化权重路径}替换为用户实际路径。
- 如果需要使用npu多卡量化，请先配置环境变量，支持多卡量化,建议双卡执行量化：
```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
```

### Atlas 300I DUO/Atlas 300I Pro/Atlas 300V稀疏量化
  - Step 1
    - 注意该量化方式仅支持在Atlas 300I DUO/Atlas 300I Pro/Atlas 300V卡上运行
    - 修改模型权重config.json中`torch_dtype`字段为`float16`
    - 请参考[msmodelslim](https://gitcode.com/Ascend/msit/blob/master/msmodelslim/docs/%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97.md)安装msModelSlim量化工具
    - 进入到msit/msmodelslim/example/Qwen的目录 `cd msit/msmodelslim/example/Qwen`；并在进入的Qwen目录下，运行量化转换脚本
    注： 安装完CANN后 需要执行source ${HOME}/Ascend/cann/set_env.sh声明ASCEND_HOME_PATH值 后续安装msmodelslim前需保证其不为空
    > 安装CANN时，如果用户未指定安装路径，则软件会安装到默认路径下，默认安装路径如下：root用户：“/usr/local/Ascend”，非root用户：“${HOME}/Ascend”，${HOME}为当前用户目录
    
    **Atlas 300I DUO**使用以下方式生成W8A8S量化权重
      ```bash
      export ASCEND_RT_VISIBLE_DEVICES=0
      export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
      python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8S量化权重路径} --calib_file ../common/cn_en.jsonl --w_bit 4 --a_bit 8 --fraction 0.011 --co_sparse True --device_type npu --use_sigma True --is_lowbit True --sigma_factor 4.0 --anti_method m4
      ```

    **Atlas 300I Pro/Atlas 300V**使用以下方式生成W8A8S量化权重
      ```bash
      python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8S量化权重路径} --calib_file ../common/cn_en.jsonl --w_bit 4 --a_bit 8 --fraction 0.011 --co_sparse True --device_type cpu --use_sigma True --is_lowbit True --sigma_factor 4.0 --anti_method m4
      ```
    > Atlas 300I Pro/Atlas 300V量化过程耗时较长，预计5小时左右，可以在Atlas 300I DUO上先生成W8A8S量化权重路径，再搬运到Atlas 300I Pro/Atlas 300V执行后续步骤。

  - Step 2：量化权重切分及压缩
    ```shell
    # 执行"jq --version"查看是否安装jq，若返回"bash：jq：command not found"，则依次执行"apt-get update"和"apt install jq"
    jq --version
    export IGNORE_INFER_ERROR=1
    cd ${llm_path}
    torchrun --nproc_per_node {TP数} -m examples.convert.model_slim.sparse_compressor --multiprocess_num 4 --model_path {W8A8S量化权重路径} --save_directory {W8A8SC量化权重路径}
    ```
    - TP数为tensor parallel并行个数
    - 注意：若权重生成时以TP=4进行切分，则运行时也需以TP=4运行
    - 示例
      ```shell
        torchrun --nproc_per_node 4 -m examples.convert.model_slim.sparse_compressor --model_path /data1/weights/model_slim/Qwen-1.5b_w8a8s --save_directory /data1/weights/model_slim/Qwen-1.5b_w8a8sc
      ```

## 纯模型推理

### 对话测试
进入llm_model路径

ATB_SPEED_HOME_PATH默认/usr/local/Ascend/llm_model,以情况而定

```shell
cd $ATB_SPEED_HOME_PATH
```

执行对话测试

```shell
torchrun --nproc_per_node 2 \
         --master_port 20037 \
         -m examples.run_pa \
         --model_path {权重路径} \
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
具体执行batch=1, 输入长度256, 输出长度256用例的2卡并行性能测试命令为：
```shell
bash run.sh pa_bf16 performance [[256,256]] 1 qwen ${weight_path} 2
```

> 注：ModelTest为大模型的性能和精度提供测试功能。使用文档请参考`${ATB_SPEED_HOME_PATH}/tests/modeltest/README.md`

>QA:报错ValueError: The path should not be a symbolic link file. 
>
>解决方法：常规snapshot_download下载权重为符号链接，可通过直接网页下载本体，替换符号链接文件。或者自行判断是否使用safe_open进行校验，此处DeepSeek-R1-Distill-Qwen-14B可直接删除base/model_test.py下safe_open使用处（459~463行）
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
"port" : 1040, #自定义
"managementPort" : 1041, #自定义
"metricsPort" : 1042, #自定义
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
"modelWeightPath" : "/data/datasets/DeepSeek-R1-Distill-Qwen-1.5B",
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

- 新建窗口测试(VLLM接口)

```shell
curl 127.0.0.1:1040/generate -d '{
"prompt": "What is deep learning?",
"max_tokens": 32,
"stream": false,
"do_sample":true,
"repetition_penalty": 1.00,
"temperature": 0.01,
"top_p": 0.001,
"top_k": 1,
"model": "qwen"
}'
```

> 注: 服务化推理的更多信息请参考[MindIE Service用户指南](https://www.hiascend.com/document/detail/zh/mindie/100/mindieservice/servicedev/mindie_service0001.html)

## 常见问题
1. ImportError: cannot import name 'shard_checkpoint' from 'transformers.modeling_utils'. 降低transformers版本可解决。

```shell
pip install transformers==4.46.3 --force-reinstall
pip install numpy==1.26.4 --force-reinstall
```