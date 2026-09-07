# MLA

Multi-head latent attention (MLA) uses the low-rank key-value joint compression to eliminate the bottleneck of key-value cache during inference, thereby supporting efficient inference. Currently, MindIE supports the single-cache MLA mechanism. The head of attention can be compressed to 1 to implement a storage- and access-friendly inference mechanism. Compared with MHA, MLA can compress 96.5% of the KV cache on the DeepSeek V2 model, greatly reducing the memory usage.

## Inference

CANN and ATB Models have been installed in the environment. For details, see *MindIE Installation Guide*.

The inference method of models supporting MLA is identical to that of other models. You can adopt the traditional LLM method during inference without setting any additional parameters.

The following uses DeepSeek-V2-Chat as an example. You can run the following commands to perform a dialog test. The inference content is "What's deep learning".

```bash
cd ${ATB_SPEED_HOME_PATH}
bash examples/models/deepseekv2/run_pa.sh {Model_weight_path}
```
