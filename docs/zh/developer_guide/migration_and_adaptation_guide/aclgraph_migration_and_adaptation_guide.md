# MindIE-LLM AclGraph 模型迁移指南

本文档面向需要在 MindIE 中迁移适配新模型的开发者，重点介绍如何把一个模型通过 aclgraph 后端接入，并完成服务化运行。下面将以 `my_model` 为例，介绍将一个新模型迁移至 MindIE 的流程。

完整的模型迁移流程包括以下步骤:

```text
1. 核心交付件 (必需)
   ├─ Router (router_my_model.py)
   ├─ Config (config_my_model.py)
   └─ Model (my_model.py)

2. 扩展实现 (可选)
   ├─ InputBuilder (input_builder_my_model.py)
   └─ ToolCallsProcessor (tool_calls_processor_my_model.py)

3. 测试验证
   └─ 加载模型权重并运行推理
```

> [!IMPORTANT]
> 
> - 如果模型需要支持 `/chat/completion` 接口、多轮对话或 ToolCall，需实现 `InputBuilder`
> - 如果模型需要支持 ToolCall 能力，需实现 `ToolCallsProcessor`

---

## 1. 创建文件

在 `mindie_llm/runtime/models/` 下创建以 `model_type` 命名的目录：

```text
mindie_llm/runtime/models/my_model/
├── __init__.py                         # 模块初始化
├── router_my_model.py                  # Router (必需)
├── config_my_model.py                  # Config (必需)
├── my_model.py                         # Model (必需)
├── input_builder_my_model.py           # InputBuilder (可选)
└── tool_calls_processor_my_model.py    # ToolCallsProcessor (可选)
```

> [!NOTE]
> **命名规范**：
>
> - 框架读取 `model_type` 字段时会统一转换成小写进行匹配，**文件夹名、文件名前缀需与 `config.json` 中的 `model_type` 字段对应**
> - 文件名统一使用小写加下划线的格式（snake_case），如 `model_type = "my_model"` → 文件路径 `my_model/my_model.py`
> - 类名使用大驼峰命名法（PascalCase），如 `MyModelRouter`、`MyModelConfig`
> [!TIP]
> 详细的匹配机制可参考 [mindie_llm/runtime/models/\_\_init\_\_.py](../../../../mindie_llm/runtime/models/__init__.py) 和 [mindie_llm/runtime/models/base/router.py](../../../../mindie_llm/runtime/models/base/router.py)。

---

## 2. 实现 Router

Router 是模型迁移的入口，负责协调 Config、Model 和其他组件的加载与初始化。

### 2.1 BaseRouter 功能

`BaseRouter` 提供了以下核心功能：

**自动加载**：

- 从 `config.json` 自动识别 `model_type`
- 动态导入对应的 Config、Model 类
- 自动加载 HuggingFace tokenizer

**可扩展接口**：

如果用户希望自定义一些行为，`BaseRouter` 提供了可扩展接口，例如：

- `_get_input_builder()`：自定义 InputBuilder
- `_get_tool_calls_parser()`：自定义工具调用解析器

### 2.2 Router 的工作流程

```text
1. BaseRouter.__post_init__()
   └─ 从 config_dict 获取 model_type

2. 访问 router.config 属性
   └─ 调用 _get_config_cls() -> 动态导入 Config 类
   └─ 创建配置实例

3. 访问 router.tokenizer 属性
   └─ 调用 _get_tokenizer() -> 加载 HuggingFace tokenizer

4. 访问 router.input_builder 属性
   └─ 调用 _get_input_builder() -> 创建 InputBuilder

5. 访问 router.model_cls 属性
   └─ 调用 _get_model_cls() -> 动态导入 Model 类
```

### 2.3 实现步骤

Router 需要继承 `BaseRouter` 。

> [!NOTE]
> **必需接口**：`__init__` - 继承基类即可，无需特殊实现
>
> **可扩展接口**（举例）：
> 
> - `_get_input_builder()` - 返回自定义的 InputBuilder 实例
> - `_get_tool_calls_parser()` - 返回工具调用解析器名称（支持 ToolsCall 时需要）

**示例**：

```python
# router_my_model.py
from dataclasses import dataclass
from mindie_llm.runtime.models.base.router import BaseRouter

@dataclass
class MyModelRouter(BaseRouter):
    """MyModel Router"""
    
    def _get_input_builder(self):
        """
        获取 InputBuilder
        
        如果模型需要特殊的输入格式（如自定义 chat_template），
        需要重写此方法返回自定义的 InputBuilder。
        默认返回 None，使用框架默认的 InputBuilder。
        """
        return None
    
    def _get_tool_calls_parser(self):
        """
        获取工具调用解析器名称
        
        如果模型支持 ToolsCall，需要重写此方法。
        默认不支持工具调用，返回 None。
        """
        return None
```

---

## 3. 实现 Config

Config 负责解析和管理模型的超参数配置。

### 3.1 Config 的作用

- 解析 HuggingFace 的 `config.json` 文件
- 提供模型超参数的访问接口
- 配置 RoPE 位置编码参数

### 3.2 基类能力

`HuggingFaceConfig` 基类中提供了常见的超参配置，若模型存在新增的超参依赖，则在 `MyModelConfig` 中新增。

> [!TIP]
> 代码参考： [huggingface_config.py](../../../../mindie_llm/runtime/config/huggingface_config.py)

### 3.3 实现步骤

Config 需要继承 `HuggingFaceConfig`。

> [!NOTE]
> **必需接口**：`__init__` - 调用父类初始化，处理模型特有的配置项
>
> **可扩展接口**（举例）：
> 
> - `_create_rope_scaling()` - 自定义 RoPE 缩放配置，详细配置请参考 [RoPE 文档](../RoPEFactoryGuide.md)

**示例**：

```python
# config_my_model.py
from dataclasses import dataclass
from mindie_llm.runtime.config.huggingface_config import HuggingFaceConfig

@dataclass
class MyModelConfig(HuggingFaceConfig):
    """MyModel 配置类，继承自 HuggingFaceConfig，定义模型特有的配置项。"""
    
    use_qk_norm: bool = True  # 是否使用 QK 归一化，如果模型有额外参数配置可类似添加
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 如果模型有新的配置项，在这里处理
    
    def _create_rope_scaling(self, rope_scaling_dict, rope_theta, max_position_embeddings):
        """创建 RoPE 缩放配置"""
        return YourRopeScaling.from_dict(
            rope_scaling_dict,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings
        )
```

---

## 4. 实现 Model

### 4.1 Model 的层次结构

MindIE-LLM 的模型按照模型结构分模块实现，可以参考下方示例：

```text
MyModelForCausalLM (顶层，包含 LM Head)
├── MyModelModel (基础模型，包含 Embedding + Layers + Norm)
│   ├── VocabParallelEmbedding (词嵌入层)
│   ├── MyModelLayer × N (Transformer 层)
│   │   ├── MyModelAttention (注意力层)
│   │   │   ├── QKVParallelLinear (QKV 投影)
│   │   │   ├── RotaryEmbedding (旋转位置编码)
│   │   │   ├── Attention (注意力计算)
│   │   │   └── RowParallelLinear (输出投影)
│   │   ├── MyModelMoe 或 MyModelMlp (MoE 或 Dense FFN)
│   │   │   ├── FusedMoE (MoE 专家网络，可选)
│   │   │   │   ├── Gate 投影
│   │   │   │   ├── Up 投影
│   │   │   │   └── Down 投影
│   │   │   ├── Gate (路由网络，MoE)
│   │   │   ├── Shared Experts (共享专家，可选)
│   │   │   └── 或 Dense MLP (Gate + Up + Down 投影)
│   │   ├── RMSNorm (输入归一化)
│   │   └── RMSNorm (注意力后归一化)
│   └── RMSNorm (最终归一化)
└── ParallelLMHead (语言模型头)
```

### 4.2 实现步骤

所有模型层都应该继承自 `nn.Module`，根据模型特性实现相应接口。每个模块的构造函数必须包含 `prefix` 参数，用于权重加载和量化配置匹配。

> [!NOTE]
> 在 forward 方法中，可以调用 layer 模块的 forward 实现，也可以使用 torch 和 torch_npu 的接口。
> [!TIP]
> **Layer 相关模块请参考**：
>
> - Linear Layer: [mindie_llm/runtime/layers/linear/](../../../../mindie_llm/runtime/layers/linear/)
> - Attention Layer: [mindie_llm/runtime/layers/attention/](../../../../mindie_llm/runtime/layers/attention/)
> - MoE Layer: [mindie_llm/runtime/layers/fused_moe/](../../../../mindie_llm/runtime/layers/fused_moe/)
> - Embedding Layer: [mindie_llm/runtime/layers/embedding/](../../../../mindie_llm/runtime/layers/embedding/)
> - Normalization Layer: [mindie_llm/runtime/layers/normalization.py](../../../../mindie_llm/runtime/layers/normalization.py)

下面给出 MyModel 标准范式：

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
        # MoE: 专家网络 + 路由网络 + 可选的共享专家
        self.experts = FusedMoE(..., prefix=f"{prefix}.experts")
        self.gate = ReplicatedLinear(..., prefix=f"{prefix}.gate")
        self.shared_experts = MyModelMLP(...)  # 可选
    
    def forward(self, hidden_states):
        ...

class MyModelLayer(nn.Module):
    def __init__(self, config, prefix, layer_idx, quant_config=None):
        super().__init__()
        self.input_layernorm = RMSNorm(...)
        self.self_attn = MyModelAttention(..., prefix=f"{prefix}.self_attn")
        self.post_attention_layernorm = RMSNorm(...)
        # MoE 模型 or Dense 模型二选一
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

> [!IMPORTANT]
> **权重加载注意事项**：
>
> 模块的权重名称优先使用 `prefix` 字段定义，如果 `prefix` 字段没有定义，则直接使用模块的属性名进行匹配。
> 
> - 权重名称与属性名一致时，无需指定 `prefix`，如 `self.o_proj = RowParallelLinear(...)`
> - 权重名称与属性名不一致时，需要指定 `prefix`，如 `qkv_proj` 对应 `q_proj, k_proj, v_proj`，此时 `prefix` 为列表：`prefix=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"]`
>
> **AclGraph 后端限制**：
>
> - 图模式**不允许**在 forward 中包含日志打印、device/stream sync 等操作

---

## 5. 实现 InputBuilder (Optional)

InputBuilder 负责处理用户输入，构建模型输入格式。如果模型有特殊的输入格式要求，或需要支持 Chat Template、Function Calling、Reasoning Mode，则需要实现 InputBuilder，InputBuilder 需要继承 `InputBuilder` 基类。

> [!NOTE]
> **必需接口**：
>
> - `__init__` - 初始化，接收 tokenizer 和可选参数
> - `_apply_chat_template()` - 应用 Chat Template

**示例**：

```python
# input_builder_my_model.py
from mindie_llm.runtime.models.base.input_builder import InputBuilder

class MyModelInputBuilder(InputBuilder):
    """MyModel InputBuilder"""
    
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
    
    def _apply_chat_template(self, conversation, tools_msg=None, **kwargs):
        """应用 Chat Template"""
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise RuntimeError("Tokenizer 不支持 apply_chat_template")
        return self.tokenizer.apply_chat_template(conversation, **kwargs)
```

---

## 6. 实现 ToolCalls 能力 (Optional)

如果模型支持 Function Calling，需要实现 ToolCallsProcessor 来解析模型输出的工具调用信息。ToolCallsProcessor 需要继承相应的基类（如 `ToolCallsProcessorWithXml`），并使用装饰器注册。

> [!NOTE]
> **必需接口**：
> 
> - `__init__` - 初始化，定义工具调用的正则表达式
> - `tool_call_start_token` - 工具调用起始标记
> - `tool_call_end_token` - 工具调用结束标记
> - `tool_call_start_token_id` - 起始标记的 Token ID
> - `tool_call_end_token_id` - 结束标记的 Token ID
> - `tool_call_regex` - 工具调用正则表达式

**示例**：

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

> [!TIP]
> 装饰器 `@ToolCallsProcessorManager.register_module(module_names=["my_model"])` 会将当前处理器注册到全局管理器中。Router 通过 `_get_tool_calls_parser()` 返回的名称（如 `"my_model"`）来查找对应的处理器。注册名称可以是一个或多个，例如 `module_names=["model_a", "model_b"]` 可让同一处理器支持多种模型。

---

## 7. 测试验证

测试验证请参考 [快速开始文档](../../user_guide/quick_start/quick_start.md)。

---

## 8. 常见问题 FAQ

### 问题 1. 如何支持分布式推理?

如果模型过大无法放入单个 Device，可以使用 Tensor Parallelism 来管理。为此，需要将模型的线性层和嵌入层替换为对应的 tensor-parallel 版本：

- `VocabParallelEmbedding`: 词表并行嵌入层
- `ParallelLMHead`: 并行语言模型头
- `RowParallelLinear`: 输入张量沿隐藏维度分片，权重矩阵沿行（输入维度）分片。在矩阵乘法后执行 all-reduce 操作以合并结果。通常用于 FFN 的第二层和注意力层的输出线性变换。
- `ColumnParallelLinear`: 输入张量复制，权重矩阵沿列（输出维度）分片。结果沿列维度分片。通常用于 FFN 的第一层和原始 Transformer 中分离的 QKV 变换。
- `MergedColumnParallelLinear`: 合并多个 `ColumnParallelLinear` 模块的列并行线性层。通常用于带加权激活函数（如 SiLU）的 FFN 第一层。
- `QKVParallelLinear`: 多头和分组查询注意力机制的查询、键和值投影的并行线性层。当键值头数小于 world size 时，该类会正确复制键值头。

> [!TIP]
> 
> - `MergedColumnParallelLinear` 支持量化方式不一致的场景，当多个并行线性层的量化方式不统一时，会变成一个包含多个 `ColumnParallelLinear` 模块的列表。
> - 框架提供 `ParallelInfoManager` 单例，用于获取并行数与 rank，通信域使用懒加载机制创建。可以通过 `get_parallel_info_manager().get(ParallelType.ATTN_TP).group_size` 获取并行信息。

更多并行层的实现细节，请参考 [mindie_llm/runtime/layers/](../../../../mindie_llm/runtime/layers/) 目录下的源码。

### 问题 2. MoE 模型如何配置专家并行?

MindIE-LLM 自动处理 MoE 的专家并行：

- 使用 `FusedMoE` 组件，框架会自动分配专家到不同设备
- 通过 `assign_experts` 函数根据并行配置分配专家
- 支持专家并行（EP）和混合并行策略
- 详细配置请参考 [DeepSeek V3.2 实现](../../../../mindie_llm/runtime/models/deepseek_v32/)

### 问题 3. 如何支持量化?

MindIE-LLM 支持 AutoQuant 能力，目前已支持的量化方式请参考 [mindie_llm/runtime/layers/quantization/](../../../../mindie_llm/runtime/layers/quantization/) 目录。使用 [msmodelslim](https://gitcode.com/Ascend/msmodelslim) 工具生成量化权重后，框架会自动识别并加载。

---

## 9. 参考资料

### 9.1 代码参考

- **DeepSeek V3.2 实现**: [mindie_llm/runtime/models/deepseek_v32/](../../../../mindie_llm/runtime/models/deepseek_v32/)
- **基类实现**: [mindie_llm/runtime/models/base/](../../../../mindie_llm/runtime/models/base/)
- **并行层实现**: [mindie_llm/runtime/layers/](../../../../mindie_llm/runtime/layers/)

建议参考 DeepSeek V3.2 的实现来了解完整的模型迁移流程。MindIE-LLM 已经支持多种模型架构，建议找到与您的模型类似的实现进行参考和适配。

### 9.2 相关文档

- **CANN 文档**: [CANN-文档-昇腾社区](https://www.hiascend.com/cann/document)
- **PTA 文档**: [Ascend Extension for PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/730/index/index.html)
- **msmodelslim 文档**: [模型量化工具](https://gitcode.com/Ascend/msmodelslim)

---

**文档版本**: v1.0  
**最后更新**: 2026-03-20  
