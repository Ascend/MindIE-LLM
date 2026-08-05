# Telechat README

星辰语义大模型 TeleChat 是由中国电信人工智能科技有限公司研发训练的大语言模型，采用 1.5 万亿 Tokens 中英文高质量语料进行训练。

- 参考实现：[TeleChat 仓库](https://github.com/Tele-AI/Telechat)

## 特性矩阵

- 设备支持情况

  | 模型及参数量 | 800I A2 | 800I A3 | 300I DUO |
  |-------------|---------|---------|----------|
  | Telechat-12B-v2 | 推荐使用 2 卡 | 推荐使用 2 卡 | 推荐使用 2 卡 |

- 支持的数据类型及量化方式

  | 模型及参数量 | FP16 | BF16 | W8A8 量化 | W8A16 量化 | KV cache 量化 | 稀疏量化 |
  |-------------|------|------|-----------|-----------|--------------|---------|
  | Telechat-12B-v2 | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ |

- 支持的部署方式

  | 模型及参数量 | 纯模型 | 服务化 |
  |-------------|-------|--------|
  | Telechat-12B-v2 | ✅ | ✅ |

## 使用说明

### 路径变量解释

| 变量名      | 含义                                                                                       |
|------------|-------------------------------------------------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录（仅源码下载场景使用）                                         |
| llm_path    | 模型仓所在路径。若使用镜像，则路径为 `/usr/local/Ascend/atb-models`；若使用 gitcode 下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径；telechat 的工作脚本所在路径为 `${llm_path}/examples/models/telechat`           |
| weight_path | 模型权重路径                                                                                 |

### 权重下载

- [Telechat-12B-v2](https://modelscope.cn/models/TeleAI/TeleChat-12B-v2/files)
- 设置模型权重加载时的数据类型

  在`{weight_path}/config.json` 文件中，添加 `torch_dtype` 字段

    ```json
    {
      ...
      "torch_dtype": "float16",
      ...
    }
    ```

### 权重转换

- 参考[此 README 文件](../../README.md)

### 量化权重转换（W8A8）

在 `llm_path` 目录下执行以下命令行

- 新增可选参数 `trust_remote_code` 代表是否信任本地的可执行文件: 默认不执行，传入此参数，则信任本地可执行文件。

``` bash
python examples/models/telechat/convert_quant_weights.py --model_path {浮点权重路径} --save_directory {W8A8 量化权重路径} --w_bit 8 --a_bit 8 --disable_level L0 --device_type cpu --anti_method m2 --act_method 3 --calib_file ${llm_path}/examples/models/telechat/boolq.jsonl --trust_remote_code
```

### 稀疏量化权重转换（W8A8SC）

请参考 [msmodelslim 安装指南](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/install_guide/install_guide.md) 安装 msModelSlim 量化工具

- Step 1

    ```shell
    cd ${llm_path}
    python examples/models/telechat/convert_quant_weights.py --model_path {浮点权重路径} --save_directory {W8A8S 量化权重路径} --w_bit 4 --a_bit 8 --calib_file ${llm_path}/examples/models/telechat/boolq.jsonl --fraction 0.011 --co_sparse True --trust_remote_code
    ```

  - 新增可选参数 `trust_remote_code` 代表是否信任本地的可执行文件: 默认不执行，传入此参数，则信任本地可执行文件。

- Step 2：量化权重切分及压缩

  ```shell
  torchrun --nproc_per_node {TP 数} -m examples.convert.model_slim.sparse_compressor --model_path {W8A8S 量化权重路径} --save_directory {W8A8SC 量化权重路径}
  ```

  - TP 数为 tensor parallel 并行个数
  - 注意：若权重生成时以 TP=4 进行切分，则运行时也需以 TP=4 运行
  - 示例

    ```shell
    torchrun --nproc_per_node 2 -m examples.convert.model_slim.sparse_compressor --model_path /data1/weights/model_slim/telechat-12b_w8a8s --save_directory /data1/weights/model_slim/telechat-12b_w8a8sc
    ```

## 服务化推理

- telechat-12b-v2 模型由于 hidden_size 变大，传 block_size 时需要更改为 96；服务化推理时需要将 `${MINDIE_LLM_HOME_PATH}/conf/config.json` 文件中的 `cacheBlockSize` 参数设置为 96。
- telechat-12b-v2 模型依赖本地模型文件；服务化推理时需要将 `${MINDIE_LLM_HOME_PATH}/conf/config.json` 文件中的 `trustRemoteCode` 参数设置为 `true`。
- 服务化推理使用详情请参考 MindIE 官网

### 精度测试

- 参考[此 README 文件](../../../tests/modeltest/README.md)

### 性能测试

- 参考[此 README 文件](../../../tests/modeltest/README.md)
