# Feature List

MindIE LLM supports foundational, quantization, long-sequence, scheduling, acceleration, and interaction features. For details on enabling each feature and its limitations, refer to the links in the overview.

<table>
    <tr>
        <th>Category</th><th>Feature</th><th>Description</th><th>Benefits</th>
    </tr>
    <tr>
        <td rowspan="8">Basic features</td><td>Multi-LoRA</td><td>Uses different LoRA weights for inference. For details, see <a href="./multi_lora.md">Multi-LoRA</a>. </td><td>Support the LoRA feature and dynamically loads and unloads weights.</td>
    </tr>
    <tr>
        <td>MoE</td><td>Enables sparse-activated expert networks to scale up model parameters without significantly increasing computational cost, thereby enhancing model capability. For details, see <a href="./moe.md">MoE</a>. </td><td>Accommodates massive knowledge with trillions of parameters, outperforming dense models in potential performance.</td>
    </tr>
    <tr>
        <td>MLA</td><td>Uses low-rank key-value joint compression to eliminate inference bottlenecks and enable efficient inference. For details, see <a href="./mla.md">MLA</a>. </td><td>Efficiently processes ultra-long contexts.</td>
    </tr>
    <tr>
        <td>Load balancing</td><td>Reduces the imbalance between NPUs, thereby improving the model inference performance. For details, see <a href="./expert_parallelism_load_balancer.md">Load balancing</a>. </td><td>Reduce delay. </td>
    </tr>
    <tr>
        <td>External shared expert</td><td>Deploys shared experts on a dedicated NPU card, separating them from routing and redundant experts. For details, see <a href="./mix_shared_routing.md">External shared expert</a>. </td><td>Optimize TPOT.</td>
    </tr>
    <tr>
        <td>Expert parallel</td><td>Deploys experts across devices to enable expert-level parallel computation. For details, see <a href="./expert_parallel.md">Expert Parallel</a>. </td><td>Reduce the latency and increase throughput.</td>
    </tr>
    <tr>
        <td>Data parallel</td><td>Batch splits inference requests across devices for parallel processing. For details, see <a href="./data_parallel.md">Data Parallel</a>. </td><td>Increase throughput. </td>
    </tr>
    <tr>
        <td>Tensor parallel</td><td>Shard tensors (e.g., weight matrices, activations) across multiple devices (e.g., NPUs) to enable distributed model inference. For details, see <a href="./tensor_parallel.md">Tensor Parallel</a>. </td><td>Reduce graphics memory per card.</td>
    </tr>
    <tr>
        <td rowspan="10">Quantization features</td><td>Anti-outlier </td><td>Suppress outliers in data to improve model quantization accuracy. For details, see <a href="./anti_outlier.md">Anti-outlier</a>. </td><td>Reduce quantization accuracy drop.</td>
    </tr>
    <tr>
        <td>PD MIX quantization</td><td>Uses different quantization modes in the prefill and decode phases of model inference. For details, see <a href="./pdmix.md">PD MIX Quantization</a>. </td><td>Reduce graphics memory.</td>
    </tr>
    <tr>
        <td>W8A8 quantization</td><td>Quantizes weights and activations to the int8 format to reduce the model size and accelerate inference computation. For details, see <a href="./w8a8.md">W8A8 Quantization</a>. </td><td>Reduce graphics memory and increase throughput.</td>
    </tr>
    <tr>
        <td>W4A8 mixed quantization</td><td>Quantizes model layers differentially: apply 4/8-bit hierarchical quantization to weights, and uniformly quantize activations to 8-bit. For details, see <a href="./w4a8_mixed_precision_quantization.md">W4A8 Mixed Quantization</a>. </td><td>Reduce graphics memory and increase throughput.</td>
    </tr>
    <tr>
        <td>W8A16 quantization</td><td>Quantizes weights to 8-bit only. For details, see <a href="./w8a16.md">W8A16 Quantization</a>. </td><td>Reduce graphics memory and increase throughput.</td>
    </tr>
    <tr>
        <td>Attention quantization</td><td>Quantizes Q, K, and V to 8-bit, effectively compressing KV cache memory, accelerating attention computation during decoding, and significantly improving model throughput. For details, see <a href="./attention_quantization.md">Attention Quantization</a>. </td><td>Reduce graphics memory and increase throughput.</td>
    </tr>
    <tr>
        <td>FA3 quantization</td><td>Quantize non-RoPE tensors of `k` to 8-bit using Attention-like quantization, while leaving RoPE tensors of `k` unquantized. This reduces KV cache memory usage and improves decoding speed, thereby increasing throughput. For details, see <a href="./fa3_quantization.md">FA3 Quantization</a>. </td><td>Reduce graphics memory and increase throughput.</td>
    </tr>
    <tr>
        <td>KV Cache Int8</td><td>Reduces memory usage and increases throughput by reducing KV memory usage and recomputation. For details, see <a href="./kv_cache_int8.md">KV Cache Int8</a>. </td><td>Reduce graphics memory and increase throughput.</td>
    </tr>
    <tr>
        <td>W8A8SC sparse quantization</td><td>Accelerates the model by zeroing out unimportant weights via sparsification, converting high-precision values to low-bit-width storage, and further reducing the weight size using compression algorithms. For details, see <a href="./w8a8sc.md">W8A8SC Sparse Quantization</a>. </td><td>Lower VRAM usage and increase maximum throughput via high sparsity.</td>
    </tr>
    <tr>
        <td>W16A16SC quantization</td><td>Sparsifies model weights via algorithm, then compresses and stores them using floating-point sparse quantization. For details, see <a href="./w16a16sc.md">W16A16SC Quantization</a>. </td><td>Use high sparsity to increase throughput and avoid dequantization.</td>
    </tr>
    <tr>
        <td rowspan="2">Long sequence features</td><td>Context parallel</td><td>Shards long sequences across the context dimension and distributes them to different devices for parallel processing, reducing Time To First Token (TTFT). For details, see <a href="./context_parallel.md">Context Parallel</a>. </td><td>Reduce memory usage and lower TTFT.</td>
    </tr>
    <tr>
        <td>Sequence parallel</td><td>Shards the KV cache so each card stores a distinct portion, reducing memory usage and enabling long-sequence support. For details, see <a href="./sequence_parallel.md">Sequence Parallel</a>. </td><td>Reduce graphics memory.</td>
    </tr>
    <tr>
        <td rowspan="3">Scheduling features</td><td>Asynchronous scheduling</td><td>For scenarios with a large `maxBatchSize` and long input/output lengths, overlaps model inference with data preparation and result return, preventing wasted NPU compute and memory resources. For details, see <a href="./asynchronous_scheduling.md">Asynchronous Scheduling</a>. </td><td>Reduce latency.</td>
    </tr>
    <tr>
        <td>SplitFuse</td><td>Breaks long prompts into smaller chunks and schedules them across multiple forward steps to reduce prefill latency. For details, see <a href="./split_fuse.md">SplitFuse</a>. </td><td>Reduce graphics memory and latency, and increase throughput</td>
    </tr>
    <tr>
        <td>SLO scheduling tuning</td><td>Improves the system throughput while ensuring the SLO. For details, see <a href="./slo_aware_scheduling_optimization.md">SLO Scheduling Tuning</a>. </td><td>Increase throughput.</td>
    </tr>
    <tr>
        <td rowspan="5">Acceleration features</td><td>Micro batch</td><td>Splits data into smaller batches during processing to fully utilize hardware resources and improve inference throughput. For details, see <a href="./micro_batch.md">Micro Batch</a>. </td><td>Increase throughput.</td>
    </tr>
    <tr>
        <td>Parallel decoding</td><td>Leverages compute capacity to mitigate memory bandwidth bottlenecks, improving compute utilization. For details, see <a href="./speculative_decoding.md">Parallel Decoding</a>. </td><td>Increase throughput.</td>
    </tr>
    <tr>
        <td>MTP</td><td>During inference, predicts multiple tokens at once rather than just the next one, significantly improving generation speed. For details, see <a href="./mtp.md">MTP</a>. </td><td>Increase throughput.</td>
    </tr>
    <tr>
        <td>Prefix cache</td><td>Reuses the KV cache corresponding to the repeated blocks across requests, reducing the prefill time. For details, see <a href="./prefix_cache.md">Prefix Cache</a>. </td><td>Lower TTFT.</td>
    </tr>
    <tr>
        <td>KV cache pooling</td><td>Integrates larger storage media—such as DRAM and even SSDs—into the prefix cache pool to break through VRAM capacity limits. For details, see <a href="./kv_cache_pool.md">KV Cache Pooling</a>. </td><td>Improve the prefix cache hit ratio.</td>
    </tr>
    <tr>
        <td rowspan="3">Interaction</td><td>Function call</td><td>Supports function calls, enabling the foundation model to use tools. For details, see <a href="./function_call.md">Function Call</a>. </td><td>Enable the use of external tools to expand the application scope.</td>
    </tr>
    <tr>
        <td>Thinking analysis</td><td>Structurally parses the output of the foundation model and separates the thinking process from the output result. For details, see <a href="./enable_reasoning.md">Thinking Analysis</a>. </td><td>Improve the inference performance in complex scenarios.</td>
    </tr>
     <tr>
        <td>Thinking budget</td><td>Controls the depth of model thinking. When the thinking content exceeds the specified thinking_budget, the system uses a prompt to truncate the thinking process. For details, see <a href="./thinking_budget.md">Thinking Budget</a>. </td><td>Improved inference performance in complex scenarios.</td>
    </tr>
    <tr>
        <td rowspan="1">Others</td><td>Offline weight partitioning</td><td>Pre-loads shard weights into tmpfs to optimize large-scale model loading and reduce NPU transfer time. For details, see <a href="./offline_weight_partitioning.md">Offline Weight Partitioning</a>. </td><td>Reduce the time required for loading weights.</td>
    </tr>
</table>

# Feature Combination Matrix

The compatibility of several features is indicated by the following symbols:

- ✅ = Fully compatible
- ❌ = Incompatible
- ❔ = To be determined

> [!NOTE]
>
> - For the cases marked with ❌ or ❔, associate them with [issues](https://gitcode.com/Ascend/MindIE-LLM/issues) for tracking.
> - Here, only mainstream models DeepSeek and Qwen are listed.

## DeepSeek models

| Feature                               | Load balancing | External deployment of shared experts | Expert parallel | Data parallel | Anti-Outlier | PD MIX quantization | W8A8 quantization | W4A8 mixed quantization | FA3 quantization | Context parallel | Sequence parallel | Asynchronous scheduling | SLO scheduling tuning | Micro batch | MTP | Prefix cache | KV cache pooling | Function call | Thinking analysis |
|:-------------------------------------:|:--------------:|:-------------------------------------:|:---------------:|:-------------:|:------------:|:-------------------:|:-----------------:|:-----------------------:|:----------------:|:----------------:|:-----------------:|:-----------------------:|:---------------------:|:-----------:|:---:|:------------:|:----------------:|:-------------:|:-----------------:|
| Load balancing                        | ✅              |                                       |                 |               |              |                     |                   |                         |                  |                  |                   |                         |                       |             |     |              |                  |               |                   |
| External deployment of shared experts | ✅              | ✅                                     |                 |               |              |                     |                   |                         |                  |                  |                   |                         |                       |             |     |              |                  |               |                   |
| Expert parallel                       | ✅              | ✅                                     | ✅               |               |              |                     |                   |                         |                  |                  |                   |                         |                       |             |     |              |                  |               |                   |
| Data parallel                         | ✅              | ✅                                     | ✅               | ✅             |              |                     |                   |                         |                  |                  |                   |                         |                       |             |     |              |                  |               |                   |
| Anti-Outlier                          | ✅              | ✅                                     | ✅               | ✅             | ✅            |                     |                   |                         |                  |                  |                   |                         |                       |             |     |              |                  |               |                   |
| PD MIX quantization                   | ✅              | ✅                                     | ✅               | ✅             | ✅            | ✅                   |                   |                         |                  |                  |                   |                         |                       |             |     |              |                  |               |                   |
| W8A8 quantization                     | ✅              | ✅                                     | ✅               | ✅             | ✅            | ✅                   | ✅                 |                         |                  |                  |                   |                         |                       |             |     |              |                  |               |                   |
| W4A8 mixed quantization               | ✅              | ✅                                     | ✅               | ✅             | ✅            | ✅                   | ✅                 | ✅                       |                  |                  |                   |                         |                       |             |     |              |                  |               |                   |
| FA3 quantization                      | ✅              | ✅                                     | ✅               | ✅             | ✅            | ✅                   | ✅                 | ❌                       | ✅                |                  |                   |                         |                       |             |     |              |                  |               |                   |
| Context parallel                      | ✅              | ✅                                     | ✅               | ❌             | ✅            | ✅                   | ✅                 | ❌                       | ✅                | ✅                |                   |                         |                       |             |     |              |                  |               |                   |
| Sequence parallel                     | ✅              | ✅                                     | ✅               | ✅             | ✅            | ✅                   | ✅                 | ✅                       | ✅                | ✅                | ✅                 |                         |                       |             |     |              |                  |               |                   |
| Asynchronous scheduling               | ✅              | ✅                                     | ✅               | ✅             | ✅            | ✅                   | ✅                 | ✅                       | ✅                | ✅                | ✅                 | ✅                       |                       |             |     |              |                  |               |                   |
| SLO scheduling tuning                 | ✅              | ✅                                     | ✅               | ✅             | ✅            | ✅                   | ✅                 | ✅                       | ✅                | ✅                | ✅                 | ✅                       | ✅                     |             |     |              |                  |               |                   |
| Micro batch                           | ✅              | ✅                                     | ✅               | ✅             | ✅            | ✅                   | ✅                 | ✅                       | ✅                | ❌                | ❌                 | ❌                       | ❌                     | ✅           |     |              |                  |               |                   |
| MTP                                   | ✅              | ✅                                     | ✅               | ✅             | ✅            | ✅                   | ✅                 | ✅                       | ✅                | ✅                | ✅                 | ✅                       | ✅                     | ✅           | ✅   |              |                  |               |                   |
| Prefix cache                          | ✅              | ✅                                     | ✅               | ✅             | ✅            | ✅                   | ✅                 | ✅                       | ✅                | ✅                | ✅                 | ✅                       | ✅                     | ❌           | ✅   | ✅            |                  |               |                   |
| KV cache pooling                      | ✅              | ✅                                     | ✅               | ✅             | ✅            | ✅                   | ✅                 | ✅                       | ✅                | ✅                | ✅                 | ✅                       | ✅                     | ❌           | ✅   | ✅            | ✅                |               |                   |
| Function call                         | ✅              | ✅                                     | ✅               | ✅             | ✅            | ✅                   | ✅                 | ✅                       | ✅                | ✅                | ✅                 | ✅                       | ✅                     | ❌           | ✅   | ✅            | ✅                | ✅             |                   |
| Thinking analysis                     | ✅              | ✅                                     | ✅               | ✅             | ✅            | ✅                   | ✅                 | ✅                       | ✅                | ✅                | ✅                 | ✅                       | ✅                     | ❌           | ✅   | ✅            | ✅                | ✅             | ✅                 |

> [!NOTE]
>
> - For the DeepSeek models, the following features can be used together: context parallel, sequence parallel, prefix cache, KV cache pooling, MTP, asynchronous scheduling, and FA3 quantization, supporting any combination of all seven. For short sequences (context length less than 16k), Context parallel and sequence parallel do not need to be enabled. For long sequences (context length 128k), the MTP feature cannot be used together with other features.

## Qwen models

| Feature                      | Multi-LoRA | Load balancing (only supported by Qwen-MoE) | Data parallel | Anti-Outlier | PD MIX quantization | W8A8 quantization | W8A16 quantization | KV Cache int8 | W8A8SC sparse quantization | W16A16SC sparse quantization | Asynchronous scheduling | SplitFuse | SLO scheduling tuning | Micro batch | Parallel decoding | Prefix cache | KV cache pooling | Function call | Thinking analysis |
|:----------------------------:|:----------:|:-------------------------------------------:|:-------------:|:------------:|:-------------------:|:-----------------:|:------------------:|:-------------:|:--------------------------:|:----------------------------:|:-----------------------:|:---------:|:---------------------:|:-----------:|:-----------------:|:------------:|:----------------:|:-------------:|:-----------------:|
| Multi-LoRA                   | ✅          |                                             |               |              |                     |                   |                    |               |                            |                              |                         |           |                       |             |                   |              |                  |               |                   |
| Load balancing               | ✅          | ✅                                           |               |              |                     |                   |                    |               |                            |                              |                         |           |                       |             |                   |              |                  |               |                   |
| Data parallel                | ✅          | ✅                                           | ✅             |              |                     |                   |                    |               |                            |                              |                         |           |                       |             |                   |              |                  |               |                   |
| Anti-Outlier                 | ❌          | ✅                                           | ✅             | ✅            |                     |                   |                    |               |                            |                              |                         |           |                       |             |                   |              |                  |               |                   |
| PD MIX quantization          | ❌          | ✅                                           | ✅             | ✅            | ✅                   |                   |                    |               |                            |                              |                         |           |                       |             |                   |              |                  |               |                   |
| W8A8 quantization            | ❌          | ✅                                           | ✅             | ✅            | ✅                   | ✅                 |                    |               |                            |                              |                         |           |                       |             |                   |              |                  |               |                   |
| W8A16 quantization           | ❌          | ✅                                           | ✅             | ✅            | ❌                   | ❌                 | ✅                  |               |                            |                              |                         |           |                       |             |                   |              |                  |               |                   |
| KV cache int8                | ❌          | ✅                                           | ✅             | ❌            | ❌                   | ✅                 | ❌                  | ✅             |                            |                              |                         |           |                       |             |                   |              |                  |               |                   |
| W8A8SC sparse quantization   | ❌          | ❌                                           | ✅             | ❌            | ❌                   | ❌                 | ❌                  | ❌             | ✅                          |                              |                         |           |                       |             |                   |              |                  |               |                   |
| W16A16SC sparse quantization | ❌          | ❌                                           | ✅             | ❌            | ❌                   | ❌                 | ❌                  | ❌             | ❌                          | ✅                            |                         |           |                       |             |                   |              |                  |               |                   |
| Asynchronous scheduling      | ❌          | ✅                                           | ✅             | ✅            | ✅                   | ✅                 | ✅                  | ✅             | ✅                          | ✅                            | ✅                       |           |                       |             |                   |              |                  |               |                   |
| SplitFuse                    | ❌          | ✅                                           | ✅             | ❌            | ✅                   | ✅                 | ❌                  | ❌             | ❌                          | ❌                            | ✅                       | ✅         |                       |             |                   |              |                  |               |                   |
| SLO scheduling tuning        | ❌          | ✅                                           | ✅             | ✅            | ✅                   | ✅                 | ✅                  | ✅             | ❌                          | ❌                            | ✅                       | ❌         | ✅                     |             |                   |              |                  |               |                   |
| Micro batch                  | ❌          | ❔                                           | ✅             | ✅            | ✅                   | ✅                 | ✅                  | ❔             | ❌                          | ❌                            | ❔                       | ❔         | ❔                     | ✅           |                   |              |                  |               |                   |
| Parallel decoding            | ❌          | ✅                                           | ✅             | ❌            | ❌                   | ✅                 | ❌                  | ❔             | ✅                          | ❌                            | ❌                       | ❌         | ❌                     | ❔           | ✅                 |              |                  |               |                   |
| Prefix cache                 | ❌          | ✅                                           | ✅             | ✅            | ✅                   | ✅                 | ✅                  | ✅             | ✅                          | ❌                            | ✅                       | ❌         | ❌                     | ❌           | ❌                 | ✅            |                  |               |                   |
| KV cache pooling             | ❌          | ✅                                           | ✅             | ✅            | ✅                   | ✅                 | ✅                  | ✅             | ✅                          | ❌                            | ✅                       | ❌         | ❌                     | ❌           | ❌                 | ✅            | ✅                |               |                   |
| Function call                | ✅          | ✅                                           | ✅             | ✅            | ✅                   | ✅                 | ✅                  | ✅             | ✅                          | ❌                            | ❔                       | ❌         | ❌                     | ❌           | ✅                 | ✅            | ✅                | ✅             |                   |
| Thinking analysis            | ✅          | ✅                                           | ✅             | ✅            | ✅                   | ✅                 | ✅                  | ✅             | ✅                          | ❔                            | ✅                       | ❌         | ✅                     | ✅           | ✅                 | ❌            | ✅                | ✅             | ✅                 |
