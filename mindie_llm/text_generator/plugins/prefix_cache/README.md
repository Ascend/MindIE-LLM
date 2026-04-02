# Prefix Cache 运行方法

## 1. Prefix Cache适用场景

    * prompt有一定长度，多次prompt请求拥有公共前缀的场景，如few-shot学习、多轮对话等

## 2. 使用服务化执行

安装MindIE，可启动MindIE Server使用服务化执行推理

### 2.1 使用llm_engine_test_new执行性能测试

执行前需要配置好config文件

#### 2.1.1 config文件配置

此处只呈现启动Prefixcache需要额外配置的字段
"BackendConfig"中的"ModelDeployConfig"中的"ModelConfig"下添加：

    "plugin_params": "{\"plugin_type\":\"prefix_cache\"}"

### 2.2 benchmark验证 mindieservice_daemon

#### 2.2.1 config配置

同2.1.1

#### 2.2.2 执行

1. 执行./mindieservice_daemon 启动服务
2. 执行benchmark（如跑MMLU数据集测试few-shot场景）

```
    benchmark --DatasetPath "/path/to/dataset/MMLU/" \
    --DatasetType "mmlu" --Http "http://127.0.0.1:1025" --ManagementHttp http://127.0.0.1:1025 \
    --ModelName "llama2_7b" --ModelPath "/home/models/llama2-7b" --TestType client --Concurrency 8 --Tokenizer True \
    --TaskKind stream --DoSampling False --TestAccuracy True --MaxOutputLen 2
```
