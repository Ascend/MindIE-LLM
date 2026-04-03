# 长序列

长序列定义为序列长度超过32K甚至可达到1M级别的文本。长序列特性的主要要求是在输入文本超长的场景下，模型回答的效果及性能也可以同时得到保障。在长序列场景下，由Attention和KV Cache部分造成的显存消耗会快速的成倍增长。因此对这部分显存的优化便是长序列特性的关键技术点。其中涉及到诸如KV Cache量化，KV多头压缩，训短推长等关键算法技术。

-  训长推长：在训练时通过较长的文本对模型的权重进行训练，从而使得模型在推理过程中对长序列输入依然可以保持良好的模型能力。
-  训短推长：模型本身通过诸如Alibi编码或序列压缩算法如NTK，YaRN等技术使得模型具备较强的自扩张能力，从而可以通过短序列训练后在长序列推理场景下获得更好的模型能力。

## 限制与约束

-  请参考[模型列表](../model_support_list.md)中的“大语言模型列表”，获取各个模型支持的序列长度。
-  目前MindIE LLM对最大序列长度的支持主要受限于以下两类因素：
    -  硬件的显存规格与模型参数量。该因素决定了在硬件允许的情况下，模型在推理时所能接受的最长输入长度。以64G  Atlas 800I A2 推理服务器硬件为例，使用8卡运行Glm4-9B-Chat模型，在显存允许的条件下能进行最长1M的长序列推理。
    -  模型权重和结构等对长序列能力的支持。该因素决定了模型在长序列场景下的生成和对话效果。对于训长推长的模型（例如Glm-4-9B-Chat-1M），MindIE LLM能保证与开源实现相同的长序列推理效果；对于训短推长的模型，MindIE LLM中实现了NTK、YaRN等相应技术，在使能相关特性后也能让模型具备与开源实现同等的长序列推理能力。需要注意的是，若想使原生仅支持短序列的模型来处理长序列输入，MindIE LLM无法保证其长序列推理结果的合理性。
    -  目前支持NTK的模型包括：Llama3。支持YaRN的模型包括：底层运行Qwen2 modeling的模型，如Qwen2, Qwen2.5，Qwen3。

## 执行推理

请兼顾硬件规格、模型参数及模型支持的最大有效推理长度来确定合适的序列长度，具体规格请参考对应模型的官方介绍文档。相较于普通推理，部分支持长序列特性的模型需要修改配置文件来支持长序列特性。以Qwen2.5-72B-Instruct为例，使能长序列特性需要修改其权重文件中的"config.json"，增加"rope\_scaling"字段（若不需要使能长序列特性，请勿添加）：

```json
{
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  // ...
  "vocab_size": 152064,
  
  // adding the following snippets
  "rope_scaling": {
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
    "type": "yarn"
  }
}
```

不同模型使能长序列特性的方式可能不同，部分模型（例如LLaMA3.1-70B-Instruct）不需要修改也可使能长序列特性。请参考各支持长序列特性模型的README，获取使能长序列特性的具体方法。

**纯模型推理场景**

模型权重使能长序列特性后，仅需要将长序列文本按照正常推理流程传入模型即可完成长序列推理。详细的执行模型推理的流程请参考[ATB Models纯模型使用](../user_manual/offline_inference.md#atb-models纯模型使用)获取更多信息和支持。

添加完长序列特性配置后，可正常执行推理。可自定义设置输入文本长度，如大于"original\_max\_position\_embeddings"的值时，可进行长序列推理。具体推理执行方式可参考以下指令：

```bash
cd ${ATB_SPEED_HOME_PATH}
torchrun --nproc_per_node [运行卡数] --master_port 20030 -m examples.run_pa --model_path [模型权重路径] --max_output_length [最大输出长度] --max_input_length [最大输入长度] --input_texts [输入文本，可支持文件或字符串]
```

> [!NOTE]说明 
> 长序列推理建议使用文本文件作为输入，如“\*.txt”。

**服务化推理场景**

模型权重使能长序列特性后，还需在服务化场景配置文件"<site-packages>/mindie_llm/conf/config.json"中配置长序列场景下支持的上下文长度。（示例中使用Batch size=1，输入127K，输出1K的场景进行配置，在实际部署中，请根据真实业务规格调整参数。）

```json
{
  "BackendConfig": {
    "ModelDeployConfig": {
      ...
      "maxInputTokenLen": 130048,
      "maxSeqLen": 131072,
    },
    "ScheduleConfig": {
      ...
      "maxBatchSize": 1,
      "maxIterTimes": 1024,
      "maxPrefillBatchSize": 1,
      "maxPrefillTokens": 130048,
    }
  }
}
```

添加完长序列特性配置后，启动服务即可。在调用接口时，可以通过curl发送包含长序列文本的请求体。
