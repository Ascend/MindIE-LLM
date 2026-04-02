# MoE

Mixture of Experts（MoE）在传统transformer结构的基础上进行了两个创新。第一个部分是用Sparse MoE layer来替换transformer结构中Feed Forward Network（FFN）。每一个FFN可扮演一个专家的角色，但针对每一个token的推理，仅需激活其中部分专家即可。这部分激活专家的筛选就涉及到了MoE的第二个关键机制：路由（routing）机制，这个Router决定了token在每一层会进入到哪一个专家。基于这两个机制的结合，MoE模型得益于其广阔的专家知识可以保证很高的模型效果，但相较于同等参数量的传统模型，它只需要激活其中部分专家，便又能同时保证其优秀的推理性能。

MoE结构的典型代表模型有Mixtral 8\*7B，Mixtral 8\*22B，DeepSeek-16B-MoE，DeepSeek-V2，DeepSeek-V3，DeepSeek-R1，Qwen3-30B-A3B，Qwen3-235B-A22B等。

## 限制与约束

能力支持特征矩阵见[表1](#table1)所示。

**表 1** **能力支持特征矩阵**  <a id="table1"></a>

|已支持模型|数据格式|量化|并行方式|硬件平台|多机多卡推理|
|--|--|--|--|--|--|
|Mixtral 8*7B|FP16|暂不支持|TP|Atlas 800I A2 推理服务器|不支持|
|Mixtral 8*22B|FP16|暂不支持|TP|Atlas 800I A2 推理服务器|不支持|
|DeepSeek-16B-MoE|FP16|暂不支持|TP|Atlas 800I A2 推理服务器|不支持|
|DeepSeek-V2|BF16|支持|TP、EP|Atlas 800I A2 推理服务器|支持|
|DeepSeek-V3|BF16|支持|TP、EP|Atlas 800I A2 推理服务器|支持|
|DeepSeek-R1|BF16|支持|TP、EP|Atlas 800I A2 推理服务器|支持|
|Qwen3-30B-A3B|BF16|支持|TP|Atlas 800I A2 推理服务器|不支持|
|Qwen3-235B-A22B|BF16|支持|TP|Atlas 800I A2 推理服务器|支持|


**模型配置参数**

模型固有参数配置请参考官方权重文件中的config.json文件。

## 执行推理

MoE类模型执行推理的方式与其他模型一致，在执行推理时您可参考传统LLM的使用方式，无需做额外配置修改。

以DeepSeek-16B-MoE为例，您可以使用以下指令执行对话测试，推理内容为"What's deep learning"。

```bash
cd ${ATB_SPEED_HOME_PATH}
bash examples/models/deepseek/run_pa_deepseek_moe.sh {模型权重路径}
```

