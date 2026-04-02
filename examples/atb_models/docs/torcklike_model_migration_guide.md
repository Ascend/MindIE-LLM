# Torch-like组图模型迁移指南
## 1. 概述
本文档介绍如何在ATB-Models中基于torch-like组图进行模型迁移。

本文档假设您了解大模型推理的基本知识，比如大模型的基本结构和推理流程。

当前已完成qwen2/qwen3/llama等模型在torch-like组图上的迁移实现。

## 2. 基本概念：

torch-like组图是ATB-Models中的一种组图方式。“组图”指调用atb算子，组成一个完整的静态计算图以供推理使用的过程。当前ATB-Models中的模型大多直接调用ATB的cpp侧接口进行组图，代码需要在cpp侧进行开发，开发效率较低。

torch-like组图将ATB的cpp侧接口包装为Python侧接口，同时引入了自动切图、组图和图融合（Fusion pass）机制，开发者可以使用类似torch的风格进行组图代码开发。以Qwen模型为例，您可以调用`atb_llm.layer`/`atb_llm.nn`中的各种torch-like api（例如`MergedColumnParallelLinear`，以`torch.nn.Module`的风格构建attention、mlp等module，在每个module的`__init__`中定义需要调用的api，然后在forward中定义如何使用这些api进行推理；接着往上构建`decoder layer`、`decoder model`以及最终的`FlashCasualLM`。

**需要注意的是**，torch-like module并非真正的`torch` module，其forward函数不能对真正的`torch.Tensor`进行计算，而是对“假Tensor”进行处理，以记录所需要调用的算子。要让torch-like module进行真实tensor运算，需要结合`atb_llm.nn.network`中的一些方法完成从torch-like module到计算图的转换。为便于开发，我们将转换FlashCausalLM的步骤封装到了`atb_llm.models.base.flash_causal_lm_v3.torch_to_mindie_graph`装饰器中，使用方法会在后续小节中说明。

## 2. 准备工作：认识torch-like API

torch-like API是我们已经开发好了的一些API接口，它们实现了对ATB算子接口的封装和扩展。

以一个简单的矩阵乘计算为例，您可以按照如下方式进行两个`torch.Tensor`的矩阵乘：
```python
import torch
from atb_llm.nn.functional import linear
from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor

# 调用算子
out = linear(Tensor("x"), Tensor("weight"), None, False, False)

# 自动构图
get_default_net().mark_output(out, "out")
matmul_engine = get_default_net().build_engine()

# 对真实Tensor进行计算
x = torch.rand(1000, 2000).to(torch.float16).npu()
weight = torch.rand(2000, 3000).to(torch.float16).npu()
inputs = {"x": x, "weight": weight}
out = torch.empty(1000, 3000).to(torch.float16).npu()
outputs = {"out": out}
matmul_engine.forward(inputs, outputs)
print(torch.allclose(torch.matmul(x, weight), out, rtol=1e-02, atol=1e-02))
```

看到这您可能有个疑问：*为什么我不直接用`torch.matmul`？*

答案是，您当然可以用`torch`完成所有计算流程，但这样（在当前的ATB框架下）无法实现将多个算子组成一个图算子，而图算子在计算效率上有很大的优势。而使用上述方式，您可以在调用`get_default_net().mark_output`和`get_default_net().build_engine`之前（即示例代码中“调用算子”部分）定义复杂的计算流程，然后只需要调用一次`get_default_net().mark_output`和`get_default_net().build_engine`，即可完成对所有算子的自动图切分和组图！我们后续的介绍都是基于这一基础概念。

除了简单的`linear`，在`atb_llm.layers`和`atb_llm.nn`中，我们已经实现了大量的torch-like API，例如在`linear`基础上包装了权重加载逻辑的`MergedColumnParallelLinear`（见[linear.py](../atb_llm/layers/linear/linear.py#L142)）、`RMSNorm`（见[normalization.py](../atb_llm/layers/norm/normalization.py#L13)等，我们非常建议您阅读它们的源码、源码的接口注释和**UT**（在[atb_models/tests/pythontest/atb_llm/nn](../tests/pythontest/atb_llm/nn/)和[atb_models/tests/pythontest/atb_llm/layers](../tests/pythontest/atb_llm/layers/)中），从而了解它们应该怎么被加到您接下来需要实现的模型迁移代码中，以及它们分别对应`torch`的哪些计算方法。
## 3. 迁移流程：必须实现的类和文件

现在我们开始正式介绍如何基于torch-like组图迁移一个模型。我们将介绍您必须完成哪些文件，以及必须实现哪些类。一个具体的例子是Qwen2，配合Qwen2的相关代码([modeling_qwen2_python.py](../atb_llm/models/qwen2/modeling_qwen2_python.py)、[flash_causal_qwen2v3.py](../atb_llm/models/qwen2/flash_causal_qwen2_v3.py))食用以下说明更佳。

### 必须实现的文件结构

```
atb_llm/models/你的模型/
|-- flash_causal_你的模型_v3.py # 注：必须由_v3.py作为后缀，且“你的模型”要与模型权重config.json中的`model_type`字段相同
|-- modeling_你的模型_python.py
|-- config_你的模型.py
|-- router_你的模型.py
```

### 必须实现的核心类
#### 1. Attention模块（如：`Qwen2Attention`）
**位置**： `modeling_你的模型_python`.py

**作用**：实现attention block，包括qkv映射、self-attention的计算、o矩阵计算等。

这里我们以一个最简单的MHA作为示例。

```python
from atb_llm import nn
from atb_llm.layers.attention.attention import Attention

class YourModelAttention(Attention):
    def __init__(
            self,
            config: BaseConfig,
            file_loader: SafetensorFileLoader,
            prefix: str,
            config_metadata: ConfigMetadata,
            infer_param: InferenceParameter,
            **kwargs
    ) -> None:
        super().__init__(config, file_loader, prefix, config_metadata, infer_param, **kwargs)
        # 初始化q、k、v、o等线性层
        self.qkv = ...
        self.dense = ...
    
    def forward(...):
        # 我们在Attention基类中已经实现了forward，如果有需要的话，您也可以重写该函数。
        q, k, v = self.qkv(...)
        ...
        q_out, k_out = nn.functional.rope(...)
        ...
        atten_score = nn.functional.paged_attention(...)
        ...
        atten_out = self.dense(...)
        return atten_out
```

#### 2. MLP模块（如：`Qwen2Mlp`）

**位置**： `modeling_你的模型_python.py`

**作用**：实现decoder layer中的mlp。

这里当然可以新增或替换为MOE，我们在这用一个简单的包含gate、up、swiglu、mul和down的mlp做一个例子。

```python
from atb_llm import nn
from atb_llm.layers.mlp.mlp import Mlp

class YourModelMlp(Mlp):
    def __init__(
            self,
            config: BaseConfig,
            file_loader: SafetensorFileLoader,
            prefix: str,
            config_metadata: ConfigMetadata,
            infer_param: InferenceParameter,
            **kwargs
    ) -> None:
        super().__init__(config, file_loader, prefix, config_metadata, infer_param, **kwargs)
        # 初始化gate up 和 down
        self.gate_up = ...
        self.down = ...
    
    def forward(...):
        # 我们在Mlp基类中已经实现了forward，如果有需要的话，您也可以重写该函数。
        gate_up_out = self.gate_up(...)
        act_out = nn.functional.activation(gate_up_out, nn.functional.ActType.SWIGLU)
        down_out = self.down(...)
        return down_out
```

#### 3. Decoder Layer模块（如`Qwen2Layer`）
**位置**： `modeling_你的模型_python.py`

**作用**：实现decoder layer，组合attention、mlp、norms。

示例：
```python
from atb_llm import nn
from atb_llm.models.base.modeling_python import BaseLayer
from atb_llm.layers.norm.normalization import RmsNorm

class YourModelLayer(BaseLayer):
    def __init__(
            self,
            config: BaseConfig,
            file_loader: SafetensorFileLoader,
            prefix: str,
            layer_idx,
            **kwargs
    ) -> None:
        super().__init__(config, file_loader, prefix, layer_idx, **kwargs)
        # 初始化attention, mlp, norm
        self.self_attn = YourModelAttention(...)
        self.mlp = YourModelMlp(...)
        self.input_layer_norm = RmsNorm(...)
        self.post_attention_layernorm = RmsNorm(...)
        
    
    def forward(...):
        norm_out = self.input_layernorm(inputs)
        attn_out = self.self_attn(...)
        res_add = attn_out + inputs
        post_norm_out = self.post_attention_layernorm(res_add)
        mlp_out = self.mlp(post_norm_out, ...)
        out = mlp_out + res_add
        return out
```

#### 4. Decoder Model模块（如`Qwen2Model`）
**位置**： `modeling_你的模型_python.py`

**作用**：实现decoder model，组合decoder layers、embeddings、final norm。

示例：
```python
from atb_llm import nn
from atb_llm.models.base.modeling_python import BaseModel
from atb_llm.layers.norm.normalization import RmsNorm
from atb_llm.layers.embedding.word_embedding import ParallelEmbedding

class YourModelModel(BaseModel): # 是的，这个名字很怪：—)
    def __init__(
            self,
            config: BaseConfig,
            file_loader: SafetensorFileLoader,
            prefix: str,
            **kwargs
    ) -> None:
        super().__init__(config, file_loader, prefix,  **kwargs)
        # 初始化embedding，layers，final norm
        self.embed_tokens = ParallelEmbedding(...)
        self.layers = nn.ModuleList(
            [
                YourModelLayer(config, file_loader, prefix, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RmsNorm(...)
    
    def forward(...):
        hidden_states = self.embed_tokens(...)
        cos_emb = nn.functional.gather(...)
        sin_emb = nn.functional.gather(...)
        for i in range(self.config.num_hidden_layers):
            hidden_states = self.layers[i](...)
        hidden_states = self.norm(...)

        return hidden_states
```

#### 5. Flash Causal模块（如`FlashQwen2ForCausalLMV3`）
**位置**： `flash_causal_你的模型.py`

**作用**：组合decoder model和lm head，加上`torch_to_mindie_graph`装饰器（一定要加）就可以对接`ModelRunner`，实现其所需的`forward`接口。

示例

```python
from atb_llm.nn.functional import gather
from atb_llm.nn.distributed import distributed as dist
from atb_llm.models.your_model.modeling_your_model_python import YourModelModel
from atb_llm.models.base.flash_causal_lm_v3 import FlashCausalLMV3, torch_to_mindie_graph
from atb_llm.layers.linear.linear import ColumnParallelLinear

@torch_to_mindie_graph()
class FlashYourModelForCausalLMV3(FlashCausalLMV3):
    def __init__(self, mindie_llm_config: MindIELLMConfig, weight_loader: SafetensorFileLoader, **kwargs) -> None:
        super().__init__(mindie_llm_config, weight_loader, **kwargs)
        self.model = YourModelModel(
            config=mindie_llm_config.hf_config,
            file_loader=weight_loader,
            mapping=mindie_llm_config.mapping,
            prefix="model",
            config_metadata=self.model_status,
            **kwargs
        )
        self.lm_head = ColumnParallelLinear(...)
    
    def forward(self, **kwargs):
        is_prefill = kwargs.get("is_prefill", True)
            
        if is_prefill:
            lm_head_indices = kwargs.get("lm_head_indices")
            hidden_states = self.model(**kwargs)
            hidden_states_ = gather(hidden_states, 0, lm_head_indices)
            lm_head_out = self.lm_head(hidden_states_)
        else:
            hidden_states = self.model(**kwargs)
            lm_head_out = self.lm_head(hidden_states)
        if self.mindie_llm_config.mapping.world_size > 1:
            logits = dist.all_gather(lm_head_out)
            logits = logits.permute([1, 0, 2])
            return {"model_out": logits}
        return {"model_out": lm_head_out}
```

#### 6. Config模块（如`Qwen2Config`）
**位置**： `config_你的模型.py`

**作用**：读取和管理huggin face权重config.json中的配置。

示例代码：

``` python
from dataclasses import dataclass
from atb_llm.models.base.config import BaseConfig


@dataclass
class YourModelConfig(BaseConfig):
    model_type = "your_model"
    # 根据huggin face权重config.json中的字段进行配置
    vocab_size = ...
    hidden_size = ...
    num_hidden_layers ...

```

#### 7. Router模块 (如`Qwen2Router`)
**位置**： `router_你的模型.py`

**作用**：让`ModelRunner`启动模型时找到模型类（FlashCausal）、config类，以及其他工具类，例如`input_builder`、`tokenizer`等。

这里给出一个最简单的示例代码。

``` python
from dataclasses import dataclass
from atb_llm.models.base.router import BaseRouter
from atb_llm.models.your_model_config import YourModelConfig


@dataclass
class YourModelRouter(BaseRouter):
    def get_config(self):
        config = YourModelConfig.from_dict(self.config_dict)
        super().check_config(config)
        return config
```

至此，就是所有需要实现的核心类了。您可以参照已实现的模型补充所需的工具类，例如`input_builder`、`tokenizer`等。

## 4. 运行和调试

按照如上说明实现了您的torch-like model后，将`atb_llm/config/config.json->llm->engine->graph`字段改为`python`，就可以进行模型调试了。你可参阅如下readme：

+ [如何运行对话测试](../examples/README.md#启动脚本)

+ [如何采集profiling进行性能调优](../README.md#性能分析)

+ [如何dump tensor以进行精度分析](../README.md#精度分析)

## 5. 扩展知识：

+ 关于`torch_to_mindie_graph`装饰器

在这个装饰器里，我们构造了一个`EngineFromNetwork`类，使其继承`FlashCausal`类，重写`forward`函数，使其自动完成首次推理时的组图、输入参数准备等工作。

`torch_to_mindie_graph`的入参是一个`*args`列表，代表需要应用到模型上的`FeatureDecorators`，例如[`SingleLoraDecorator`](../atb_llm/models/base/feature_decorator/v3/singlelora_decorator.py)和[`MultiLoraDecorator`](../atb_llm/models/base/feature_decorator/v3/multilora_decorator.py)。我们设计`FeatureDecorator`和`torch_to_mindie_graph`的目的是实现多特性自动叠加（因为某些特性在ATB Model静态图框架下，需要创建额外的计算图。如果多个复杂特性叠加起来，就需要创建叠加图）。

+ 如果我的模型有一个特殊的输入/入参，怎么在torch-like组图上迁移？

FlashCausalLMV3预留了一个接口：`_update_model_inputs`，其返回值为三个字典（基类中返回三个空字典），分别用于更新静态图执行运算时所需的`engine_inputs`, `engine_outputs`, `engine_runtime_params`，您可阅读FlashCausalLMV3和`torch_to_mindie_graph`的代码，了解这三个字典分别代表什么。如果您需要添加特殊的输入，只需要重写`FlahsCausal`的`_update_model_inputs`，在对应字典中加入所需要更新的键值对即可，如：

``` python
def _update_model_inputs(self, input_metadata):
    your_special_input: torch.Tensor = ...
    return {"your_special_input", your_special_input}, {}, {}
```