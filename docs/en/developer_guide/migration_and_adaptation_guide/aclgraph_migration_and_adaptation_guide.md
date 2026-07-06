# MindIE-LLM AclGraph Model Porting Guide

This document is intended for developers who need to port and adapt new models to MindIE. It focuses on how to integrate a model using the aclgraph backend and run it as a service. The following sections use `my_model` as an example to walk through the process for porting a new model to MindIE.

The complete model porting process includes the following steps:

```text
1.(Mandatory) Core deliverables
   ├─ Router (router_my_model.py)
   ├─ Config (config_my_model.py)
   └─ Model (my_model.py)

2. (Optional) Extension implementation
   ├─ InputBuilder (input_builder_my_model.py)
   └─ ToolCallsProcessor (tool_calls_processor_my_model.py)

3. Test and verification
   └─ Load the model weights and run inference.
```

> [!NOTE]
>
> - If the model needs to support the `/chat/completion` API, multi-turn dialog, or ToolCall, implement `InputBuilder`.
> - If the model needs to support the ToolCall capability, implement `ToolCallsProcessor`.

---

## 1. Creating a File

Create a directory named `model_type` under `mindie_llm/runtime/models/`:

```text
mindie_llm/runtime/models/my_model/
├── __init__.py                         # Module initialization
├── router_my_model.py                  # (Mandatory) Router
├── config_my_model.py                  # (Mandatory) Config
├── my_model.py                         # (Mandatory) Model
├── input_builder_my_model.py           # (Optional) InputBuilder
└── tool_calls_processor_my_model.py    # (Optional) ToolCallsProcessor
```

> [!NOTE]
> **Naming Conventions**
>
> - When reading the `model_type` field, the framework converts it to lowercase for matching. **The folder name and file name prefix must match the `model_type` field in `config.json`.**
> - File names must be in lowercase with underscores (snake_case), for example, `model_type = "my_model"` → file path `my_model/my_model.py`.
> - Class names use PascalCase, for example, `MyModelRouter` and `MyModelConfig`.
>
>   For details about the matching mechanism, see [mindie_llm/runtime/models/\_\_init\_\_.py](../../../../mindie_llm/runtime/models/__init__.py) and [mindie_llm/runtime/models/base/router.py](../../../../mindie_llm/runtime/models/base/router.py).

## 2. Implementing the Router

The Router is the entry point for model porting, coordinating the loading and initialization of the Config, Model, and other components.

### 2.1 BaseRouter Functions

`BaseRouter` provides the following core functions:

**Automatic loading**:

- Automatically identifies `model_type` from `config.json`.
- Dynamically imports the corresponding Config and Model classes.
- Automatically loads the HuggingFace tokenizer.

**Extensible interfaces:**:

To customize behaviors, `BaseRouter` provides extensible interfaces, for example:

- `_get_input_builder()`: custom InputBuilder
- `_get_tool_calls_parser()`: custom tool call parser

### 2.2 Router Workflow

```text
1. BaseRouter.__post_init__()
   └─Obtain model_type from config_dict.

2. Access the router.config attribute.
   └─ Call _get_config_cls() -> Dynamically import the Config class.
   └─ Create a configuration instance.

3. Access the router.tokenizer attribute.
   └─ Call _get_tokenizer() -> Load the HuggingFace tokenizer.

4. Access the router.input_builder attribute.
   └─ Call _get_input_builder() -> Create an InputBuilder.

5. Access the router.model_cls attribute.
   └─ Call _get_model_cls() -> Dynamically import the Model class.
```

### 2.3 Implementation Procedure

The Router needs to inherit `BaseRouter`.

> [!NOTE]
> **Mandatory API**: `__init__` - Inherit from the base class, no special implementation needed.
>
> **Extensible APIs** (examples):
>
> - `_get_input_builder()` - Returns the custom InputBuilder instance.
> - `_get_tool_calls_parser()` - Returns the name of the tool call parser (required when ToolsCall is supported).

**Example:**

```python
# router_my_model.py
from dataclasses import dataclass
from mindie_llm.runtime.models.base.router import BaseRouter

@dataclass
class MyModelRouter(BaseRouter):
    """MyModel Router"""

    def _get_input_builder(self):
        """
        Obtain InputBuilder.

        If the model requires a special input format (for example, a custom chat_template),
        override this method to return the custom InputBuilder.
        By default, None is returned, and the default InputBuilder of the framework is used.
        """
        return None

    def _get_tool_calls_parser(self):
        """
        Obtain the name of the tool call parser.

        If the model supports ToolsCall, override this method.
        By default, tool call is not supported, and None is returned.
        """
        return None
```

## 3. Implementing Config

Config parses and manages the hyperparameter configuration of the model.

### 3.1 Functions of Config

- Parses the `config.json` file of Hugging Face.
- Provides an interface for accessing model hyperparameters.
- Configures RoPE positional encoding parameters.

### 3.2 Base Class Capabilities

The `HuggingFaceConfig` base class provides common hyperparameter configurations. New hyperparameter dependencies must be added to `MyModelConfig`.

> [!NOTE]
> Code reference: [huggingface_config.py](../../../../mindie_llm/runtime/config/huggingface_config.py)

### 3.3 Implementation Procedure

Config needs to inherit `HuggingFaceConfig`.

> [!NOTE]
> **Mandatory API**: `__init__` - Calls parent class initialization to handle model-specific configuration items.
>
> **Extensible APIs** (examples):
>
> - `_create_rope_scaling()` - Customizes the RoPE scaling configuration. For details, see the [RoPE documentation](../architecture_design/RoPEFactoryGuide.md).

**Example:**

```python
# config_my_model.py
from dataclasses import dataclass
from mindie_llm.runtime.config.huggingface_config import HuggingFaceConfig

@dataclass
class MyModelConfig(HuggingFaceConfig):
    """MyModel configuration class, which is inherited from HuggingFaceConfig and defines model-specific configuration items."""

    use_qk_norm: bool = True  # Whether to use QK normalization. Additional model parameters can be added in a similar manner.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Process new configuration items here.

    def _create_rope_scaling(self, rope_scaling_dict, rope_theta, max_position_embeddings):
        """Create a RoPE scaling configuration."""
        return YourRopeScaling.from_dict(
            rope_scaling_dict,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings
        )
```

## 4. Implementing a Model

### 4.1 Model Hierarchy

The MindIE-LLM model is implemented by modules based on the model structure. The following is an example:

```text
MyModelForCausalLM (top layer, including the LM head)
├── MyModelModel (base model, including Embedding + Layers + Norm)
│   ├── VocabParallelEmbedding (word embedding layer)
│   ├── MyModelLayer × N (Transformer layer)
│   │   ├── MyModelAttention (attention layer)
│   │   │   ├── QKVParallelLinear (QKV projection)
│   │   │   ├── RotaryEmbedding (rotary positional encoding)
│   │   │   ├── Attention (attention computation)
│   │   │   └── RowParallelLinear (output projection)
│   │   ├── MyModelMoe or MyModelMlp (MoE or dense FFN)
│   │   │   ├── FusedMoE (optional MoE experts)
│   │   │   │   ├── Gate projection
│   │   │   │   ├── Up projection
│   │   │   │   └── Down projection
│   │   │   ├── Gate (MoE router)
│   │   │   ├── Shared Experts (optional)
│   │   │   └──or Dense MLP (Gate + Up + Down projections)
│   │   ├── RMSNorm (input normalization)
│   │   └── RMSNorm (post-attention normalization)
│   └── RMSNorm (final normalization)
└── ParallelLMHead (language model head)
```

### 4.2 Implementation Procedure

All model layers should inherit from `nn.Module` and implement corresponding APIs based on model features. The constructor of each module must contain the `prefix` parameter, which is used for weight loading and quantization configuration matching.

> [!NOTE]
> For the `forward` method, you can either call the layer module's `forward` implementation or use `torch` and `torch_npu` APIs.
>
> **For details about layer-related modules, see:**
>
> - Linear Layer: [mindie_llm/runtime/layers/linear/](../../../../mindie_llm/runtime/layers/linear/)
> - Attention Layer: [mindie_llm/runtime/layers/attention/](../../../../mindie_llm/runtime/layers/attention/)
> - MoE Layer: [mindie_llm/runtime/layers/fused_moe/](../../../../mindie_llm/runtime/layers/fused_moe/)
> - Embedding Layer: [mindie_llm/runtime/layers/embedding/](../../../../mindie_llm/runtime/layers/embedding/)
> - Normalization Layer: [mindie_llm/runtime/layers/normalization.py](../../../../mindie_llm/runtime/layers/normalization.py)

The following is the standard `MyModel` pattern:

```python
class MyModelAttention(nn.Module):
    def __init__(self, config, prefix, quant_config=None):
        super().__init__()
        self.qkv_proj = QKVParallelLinear(..., prefix=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"])
        self.o_proj = RowParallelLinear(...)
        self.rope_emb = get_rope(...)
        self.attn = Attention(...)

    def forward(self, positions, hidden_states):
        ...

class MyModelMoe(nn.Module):
    def __init__(self, config, prefix, quant_config=None):
        super().__init__()
        # MoE: expert network + routing network + optional shared experts
        self.experts = FusedMoE(..., prefix=f"{prefix}.experts")
        self.gate = ReplicatedLinear(..., prefix=f"{prefix}.gate")
        self.shared_experts = MyModelMLP(...)  # Optional

    def forward(self, hidden_states):
        ...

class MyModelLayer(nn.Module):
    def __init__(self, config, prefix, layer_idx, quant_config=None):
        super().__init__()
        self.input_layernorm = RMSNorm(...)
        self.self_attn = MyModelAttention(..., prefix=f"{prefix}.self_attn")
        self.post_attention_layernorm = RMSNorm(...)
        # MoE model or dense model
        self.mlp = MyModelMoe(..., prefix=f"{prefix}.mlp") # MOE
        self.mlp = MyModelMlp(..., prefix=f"{prefix}.mlp") # DENSE

    def forward(self, positions, hidden_states):
        ...

class MyModelModel(nn.Module):
    def __init__(self, config, prefix, quant_config=None):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(...)
        self.layers = nn.ModuleList([
            MyModelLayer(..., prefix=f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(...)

    def forward(self, input_ids, positions):
        ...

class MyModelForCausalLM(BaseModelForCausalLM):
    def __init__(self, mindie_llm_config):
        super().__init__(mindie_llm_config)
        self.model = MyModelModel(..., prefix="model")
        self.lm_head = ParallelLMHead(..., prefix="lm_head")

    def forward(self, input_ids, positions, ...):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        ...
```

> [!NOTE]
> **Precautions for weight loading**:
>
> The weight name of a module is preferentially defined by the `prefix` field. If the `prefix` field is not defined, the attribute name of the module is used for matching.
>
> - If the weight name is the same as the attribute name, `prefix` is not required, for example, `self.o_proj = RowParallelLinear(...)`.
> - If the weight name is different from the attribute name, `prefix` needs to be specified. For example, `qkv_proj` maps to `q_proj, k_proj, v_proj`, so `prefix` is a list: `prefix=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"]`.
>
> **AclGraph backend restrictions**:
>
> - In graph mode, operations such as log printing and device/stream sync **are not allowed** in `forward`.

---

## 5. (Optional) Implementing `InputBuilder`

InputBuilder processes user input and constructs the model input format. If your model requires a custom input format, or needs to support features such as Chat Template, Function Calling, or Reasoning Mode, implement `InputBuilder` by inheriting from the base `InputBuilder` class.

> [!NOTE]
> **Mandatory APIs**:
>
> - `__init__` - Performs initialization and receives tokenizer and optional parameters.
> - `_apply_chat_template()` - Applies Chat Template.

**Example:**

```python
# input_builder_my_model.py
from mindie_llm.runtime.models.base.input_builder import InputBuilder

class MyModelInputBuilder(InputBuilder):
    """MyModel InputBuilder"""

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)

    def _apply_chat_template(self, conversation, tools_msg=None, **kwargs):
        """Apply Chat Template"""
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise RuntimeError("Tokenizer does not support apply_chat_template.")
        return self.tokenizer.apply_chat_template(conversation, **kwargs)
```

## 6. (Optional) Implementing the ToolCalls Capability

If the model supports Function Calling, implement `ToolCallsProcessor` to parse the tool call information output by the model. `ToolCallsProcessor` needs to inherit the corresponding base class (such as `ToolCallsProcessorWithXml`) and be registered using a decorator.

> [!NOTE]
> **Mandatory APIs**:
>
> - `__init__` - Performs initialization and defines the regular expression for tool call.
> - `tool_call_start_token` - tool call start flag
> - `tool_call_end_token` - tool call end flag
> - `tool_call_start_token_id` - token ID of the start flag
> - `tool_call_end_token_id` - token ID of the end flag
> - `tool_call_regex` - regular expression for tool call

**Example:**

```python
# tool_calls_processor_my_model.py
import re
from mindie_llm.runtime.models.base.tool_calls_processor import (
    ToolCallsProcessorWithXml, ToolCallsProcessorManager
)

@ToolCallsProcessorManager.register_module(module_names=["my_model"])
class ToolCallsProcessorMyModel(ToolCallsProcessorWithXml):
    """MyModel ToolCallsProcessor"""

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._tool_calls_regex = re.compile(r'<tool_call\s*({.*?})\s*/>', re.DOTALL)

    @property
    def tool_call_start_token(self) -> str:
        return "<tool_call"

    @property
    def tool_call_end_token(self) -> str:
        return "/>"

    @property
    def tool_call_start_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids("<tool_call")

    @property
    def tool_call_end_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids("/>")

    @property
    def tool_call_regex(self):
        return self._tool_calls_regex
```

> [!NOTE]
> The `@ToolCallsProcessorManager.register_module(module_names=["my_model"])` decorator registers the current processor with the global manager. The router searches for the corresponding processor based on the name (for example, `"my_model"`) returned by `_get_tool_calls_parser()`. There can be one or more registration names. For example, `module_names=["model_a", "model_b"]` allows the same processor to support multiple models.

## 7. Test and verification

For details about the test and verification, see [Quick Start](../../user_guide/quick_start/quick_start.md).

## 8. FAQs

### Question 1: How is distributed inference supported?

If the model is too large to fit on a single device, you can use tensor parallelism to manage it. To do this, you need to replace the linear and embedding layers of the model with their tensor-parallel counterparts:

- `VocabParallelEmbedding`: vocabulary-parallel embedding layer.
- `ParallelLMHead`: parallel language model head.
- `RowParallelLinear`: shards input tensor by hidden dimensions, and shards the weight matrix by rows (input dimension). The all-reduce operation is performed after matrix multiplication to combine the results. It is typically used for the output linear transformation of the second layer of the FFN and the attention layer.
- `ColumnParallelLinear`: replicates the input tensor, shards the weight matrix by columns (output dimension), and shards the result by columns. It is typically used for QKV transformation of the first layer of the FFN and the original Transformer.
- `MergedColumnParallelLinear`: combines multiple `ColumnParallelLinear` modules. It is typically used for the first layer of the FFN with a weighted activation function (such as SiLU).
- `QKVParallelLinear`: provides the query, key, and value projections for multi-head and grouped-query attention. When the number of key-value heads is less than the world size, this class correctly copies the key-value heads.

> [!NOTE]
>
> - `MergedColumnParallelLinear` supports mixed quantization schemes. When the quantization modes of multiple parallel linear layers are different, it falls back to a list of `ColumnParallelLinear` modules.
> - The framework provides a `ParallelInfoManager` singleton to obtain the number of parallel size and ranks. The communication domain is created using the lazy loading mechanism. You can obtain parallel information through `get_parallel_info_manager().get(ParallelType.ATTN_TP).group_size`.

For more details about the implementation of parallel layers, see the source code in the [mindie_llm/runtime/layers/](../../../../mindie_llm/runtime/layers/) directory.

### **Question 2:** How is the expert parallelism configured for MoE models?

MindIE-LLM automatically handles MoE expert parallelism:

- Uses the `FusedMoE` component to distribute experts to different devices.
- Assigns experts based on parallel settings through `assign_experts`.
- Supports expert parallelism (EP) and hybrid parallelism strategies.
- For detailed configuration, see [DeepSeek V3.2 Implementation](../../../../mindie_llm/runtime/models/deepseek_v32/).

### Question 3: How can quantization be supported?

MindIE-LLM supports the AutoQuant capability. For details about the currently supported quantization modes, see the [mindie_llm/runtime/layers/quantization/](../../../../mindie_llm/runtime/layers/quantization/) directory. After the [msmodelslim](https://gitcode.com/Ascend/msmodelslim) tool is used to generate quantized weights, the framework automatically identifies and loads them.

## 9. References

### 9.1 Code Reference

- **DeepSeek V3.2 implementation**: [mindie_llm/runtime/models/deepseek_v32/](../../../../mindie_llm/runtime/models/deepseek_v32/)
- **Base class implementation**: [mindie_llm/runtime/models/base/](../../../../mindie_llm/runtime/models/base/)
- **Parallel layer implementation**: [mindie_llm/runtime/layers/](../../../../mindie_llm/runtime/layers/)

You are advised to refer to DeepSeek V3.2 implementation to understand the complete model porting process. MindIE-LLM supports multiple model architectures. You are advised to find an implementation similar to your model for reference.

### 9.2 Related Documents

- **CANN**: [Documentation community edition](https://www.hiascend.com/cann/document)
- **PyTorch**: [Ascend Extension for PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/730/index/index.html)
- **msmodelslim**: [Model Quantization Tool](https://gitcode.com/Ascend/msmodelslim)
