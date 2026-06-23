# Long Sequence

A long sequence is defined as a text whose sequence length exceeds 32K or even reaches 1M. The primary goal of the long sequence feature is to ensure that the model's answering effectiveness and performance are maintained, even when the input text is excessively long. In long sequence scenarios, the memory consumed by attention and KV cache increases exponentially. Therefore, optimizing the graphics memory is the key to the long sequence feature. Key algorithmic technologies include KV cache quantization, KV multi-header compression, and short-sequence training with long-sequence inference.

- Long-sequence training/inference: During training, a long text is used to train weights of a model, so that the model can still maintain a good capability for long-sequence input in an inference process.
- Short-sequence training and long-sequence inference: A model uses technologies such as ALiBi encoding or sequence compression algorithms (such as NTK and YaRN) to ensure a strong auto-scale capability. In this way, the model can obtain a better capability in long-sequence inference phase after short-sequence training.

## Constraints

- For details about the sequence length supported by each model, see the "Large Language Model List" in [Model List](../model_support_list.md).
- The maximum sequence length supported by MindIE LLM is determined by the following factors:
    - Specifications of hardware graphics memory and the number of model parameters: This determines the maximum input length that the model can accept during inference, given the limits of the hardware. For example, for the Atlas 800I A2 inference server of 64 GB, if the Glm4-9B-Chat model is running on eight cards, up to 1M long sequence can be inferred with sufficient graphics memory.
    - Model weights and structure: This determines the generation and dialog performance of the model in the long sequence scenario. For a long-sequence training and inference model (such as Glm-4-9B-Chat-1M), MindIE LLM ensures the same long-sequence inference effect as open-source models. For a short-sequence training and long-sequence inference model, MindIE LLM leverages technologies such as NTK and YaRN (with related features enabled) to ensure the same long-sequence inference capability as open-source models. Note that if you want to use a model that natively supports only short sequences to process long sequence input, MindIE LLM cannot ensure the rationality of the long sequence inference output.
    - Currently, NTK is supported by Llama3. YaRN is supported by models running Qwen2 modeling, such as Qwen2, Qwen2.5, and Qwen3.

## Inference

Determine a proper sequence length based on the hardware specifications, model parameters, and the maximum valid inference length supported by a model. For details about the specifications, see the official documentation of the corresponding model. Unlike common inference, some models that support the long sequence feature require modification of configuration files to enable this feature. For example, to enable the long sequence feature for Qwen2.5-72B-Instruct, you need to add the `rope_scaling` field to `config.json` in the weight file. (If the long sequence feature is not required, do not add this field.)

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

The methods of enabling the long sequence feature vary according to models. For some models (such as LLaMA3.1-70B-Instruct), the feature can be enabled without any modification. For details about how to enable the long sequence feature, see the README file of each model that supports this feature.

**Pure model inference**

After the long sequence feature is enabled for the model weights, simply transfer the long sequence text to the model following the standard inference process to complete long sequence inference. For details about the model inference process, see [ATB Models: Pure Model Usage](../user_manual/offline_inference.md#atb-models-pure-model-usage).

After the configurations of the long sequence feature are added, the inference can be performed properly. You can customize the input text length. If the length exceeds the value of `original_max_position_embeddings`, long sequence inference can be performed. The following commands are an example:

```bash
cd ${ATB_SPEED_HOME_PATH}
torchrun --nproc_per_node [Number_of_running_cards] --master_port 20030 -m examples.run_pa --model_path [Model_weight_path] --max_output_length [Maximum_output_length] --max_input_length [Maximum_input_length] --input_texts [Input_text, which can be a file or character string]
```

> [!NOTE]NOTE
> You are advised to use a text file (such as `*.txt`) as the input for long sequence inference.

**Inference serving**

Once the model weights have been configured to enable long sequence support, you need to configure the context length supported in the long sequence scenario in the configuration file `<site-packages>/mindie_llm/conf/config.json` in the serving scenario. (In the following example, the configuration is based on the scenario where the batch size is 1, the input is 127K long, and the output is 1K long. Adjust the parameters based on the actual service specifications.)

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

After the configurations of the long sequence feature are added, start the service. When calling an API, you can use `curl` to send a request body containing long sequence text.
