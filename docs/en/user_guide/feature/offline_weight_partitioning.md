# Offline Weight Sharding

During weight loading, MindIE fully loads the weight files in `safetensors` format by default, shards the weights in memory based on the parallelism policy, and then transfers the weights to the NPU in host-to-device (H2D) mode. For large-scale parameter models such as DeepSeek, an offline weight sharding policy can be used to reduce the weight loading time. Weights are sharded in advance based on the runtime parallelism policy and stored in tmpfs to implement efficient loading.

## Constraints

- Only the DeepSeek-R1 and DeepSeek-V3 models support this feature.
- The configuration used for offline weight sharding must be consistent with the configuration used during model inference.
- This feature is supported only in the Atlas 800I A2 inference server two-node cluster scenario and the Atlas 800I A3 SuperPoD server single-node scenario.
- This feature cannot be enabled together with the shared expert and routed expert merging feature.
- This feature cannot be enabled together with the dynamic load balancing feature.

## Generating Weights

The following uses the Atlas 800I A3 SuperPoD server single-node scenario as an example. You can use the following script to shard the weights.

```bash
# To enable the MTP weight in the online serving inference scenario, set the following environment variables:
export DEEPSEEK_MTP=1
# Weight Sharding
torchrun --nproc_per_node 16 --master_port 20030 -m examples.convert.weight_sharder --model_path *{Path_to_the_complete_set_of_weights}* --dp 2 --tp 8 --moe_tp 4 --moe_ep 4 --save_directory *{Path_to_the_sharded_weights}*
```

The following uses dual Atlas 800I A2 inference servers as an example. You can use the following script to shard weights.

```bash
# To enable the MTP weight in the online serving inference scenario, set the following environment variables:
export DEEPSEEK_MTP=1
export RANK_TABLE_FILE={Ranktable_file_path}
# Weight Sharding
torchrun --nnodes=2 --nproc_per_node 8 --node_rank=0 --master_addr="IP address of the master node" --master_port 20030 -m examples.convert.weight_sharder --model_path *{Path_to_the_complete_set_of_weights}* --dp 2 --tp 8 --moe_tp 4 --moe_ep 4 --save_directory *{Path_to_the_sharded_weights}*

torchrun --nnodes=2 --nproc_per_node 8 --node_rank=1 --master_addr="IP address of the master node" --master_port 20030 -m examples.convert.weight_sharder --model_path *{Path_to_the_complete_set_of_weights}* --dp 2 --tp 8 --moe_tp 4 --moe_ep 4 --save_directory *{Path_to_the_sharded_weights}*
```

Weight directory structure after sharding:

```text
├── config.json
├── configuration.json
├── generation_config.json
├── model-000
│   └── model.safetensors
...
├── model-015
│   └── model.safetensors
├── model-attn-tp-000
│   └── model.safetensors
...
├── model-attn-tp-007
│   └── model.safetensors
├── model-dense-tp-000
│   └── model.safetensors
...
├── model-dense-tp-007
│   └── model.safetensors
├── model-moe-tp-000-ep-000
│   ├── model-00001-of-00005.safetensors
│   ├── model-00002-of-00005.safetensors
│   ├── model-00003-of-00005.safetensors
│   ├── model-00004-of-00005.safetensors
│   └── model-00005-of-00005.safetensors
...
├── model-moe-tp-003-ep-003
│   ├── model-00001-of-00005.safetensors
│   ├── model-00002-of-00005.safetensors
│   ├── model-00003-of-00005.safetensors
│   ├── model-00004-of-00005.safetensors
│   └── model-00005-of-00005.safetensors
├── model-norm
│   └── model.safetensors
├── model_sharded_metadata.json
├── quant_model_description_w8a8_dynamic.json
├── tokenizer.json
└── tokenizer_config.json
```

> [!NOTE]
>
>- After sharding, the model weights are stored in different directories based on the model layer, norm module, attention module, dense module, and moe module.
>- After sharding, the `model_sharded_metadata.json` file is created to index the sharding policy and sharded files.

## Inference

The online serving inference scenario is used as an example.

1. Open the `config.json` file of the server.

   - **Installation using the `.whl` package:**

    ```bash
    cd {MindIE_installation_directory}/mindie_llm/
    vi conf/config.json
    ```

   - **Installation using the `.run` package:**

    ```bash
    cd {MindIE_installation_directory}/latest/mindie-service
    vi conf/config.json
    ```

2. Set serving parameters. Change the model weight path to the path where the sharded weight files are stored. For details about the serving parameters, see [Configuration Parameters (Serving)](../user_manual/service_parameter_configuration.md). The following is an example of parameter configuration:

    ```json
    "ModelDeployConfig" :
    {
       "maxSeqLen" : 2560,
       "maxInputTokenLen" : 2048,
       "truncation" : 0,
       "ModelConfig" : [
         {
             "modelInstanceType" : "Standard",
             "modelName" : "DeepSeek-R1_w8a8",
             "modelWeightPath" : "Path to the sharded weight files",
             "worldSize" : 16,
             "cpuMemSize" : 5,
             "npuMemSize" : -1,
             "backendType" : "atb",
             "trustRemoteCode" : false
          }
       ]
    }
    ```
