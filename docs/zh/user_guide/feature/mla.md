# MLA

MLA（Multi-head Latent Attention），利用低秩键值联合压缩来消除推理时键值缓存的瓶颈，从而支持高效推理。当前MindIE支持单Cache的MLA机制，可以将Attention的head压缩为1，实现存储和访存友好的推理机制。相比MHA实现，MLA在DeepSeek V2模型上可以压缩96.5%的KV Cache，极大节省显存占用量。

## 执行推理

已在环境上安装CANN和ATB Models详情请参见《MindIE安装指南》。

支持MLA的模型执行推理的方式与其他模型一致，在执行推理时您可参考传统LLM的使用方式，无需做额外配置修改。

以DeepSeek-V2-Chat为例，您可以使用以下指令执行对话测试，推理内容为"What's deep learning"。

```bash
cd ${ATB_SPEED_HOME_PATH}
bash examples/models/deepseekv2/run_pa.sh {模型权重路径}
```

