import unittest
from unittest.mock import patch, MagicMock
import torch
from torch import nn
from mindie_llm.runtime.models.deepseek_v3.deepseek_v3 import DeepseekV3Layer, DeepseekV3Model, \
    DeepseekV3Moe, DeepseekV3MLP, Indexer, DeepseekV3Attention, DeepseekV3ForCausalLM
from mindie_llm.runtime.models.deepseek_v3.config_deepseek_v3 import DeepseekV3Config
from mindie_llm.runtime.utils.distributed import set_parallel_info_manager
from mindie_llm.runtime.layers.quantization.ms_model_slim.quantization_config import QuantizationConfig
from mindie_llm.runtime.layers.quantization.ms_model_slim.quant_type import QuantType, InferenceMode
from mindie_llm.runtime.layers.normalization import RMSNorm, LayerNorm
from mindie_llm.runtime.layers.linear.linear import ReplicatedLinear, ColumnParallelLinear, \
    RowParallelLinear, MergedColumnParallelLinear
from mindie_llm.runtime.layers.embedding.embedding import VocabParallelEmbedding, ParallelLMHead
from mindie_llm.runtime.model_runner.forward_context import set_forward_context
from mindie_llm.runtime.layers.fused_moe.fused_moe import FusedMoE
from mindie_llm.runtime.layers.parameter import BaseParameter


config_dict = {
    "q_lora_rank": 1536,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "kv_lora_rank": 512,
    "v_head_dim": 128,
    "n_routed_experts": 64,
    "n_shared_experts": 2,
    "num_experts_per_tok": 8,
    "first_k_dense_replace": 1,
    "moe_layer_freq": 1,
    "num_hidden_layers": 28,
    "topk_method": "greedy",
    "topk_group": 1,
    "n_group": 1,
    "rope_scaling": {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 40,
    "mscale": 0.707,
    "mscale_all_dim": 0.707,
    "original_max_position_embeddings": 4096,
    "type": "yarn",
    "parallel_embedding": True
    }
}
config = DeepseekV3Config.from_dict(config_dict)
config.routed_scaling_factor = 2
config.hidden_size = 7168
config.vocab_size = 50000
config.rms_norm_eps = 1e-8
config.max_position_embeddings = 2048
config.torch_dtype = torch.bfloat16
config.num_attention_heads = 128
config.qk_nope_head_dim = 128
config.qk_rope_head_dim = 64
config.qk_head_dim = 192
config.index_n_heads = 64
config.index_head_dim = 128
config.q_lora_rank = 1536
config.intermediate_size = 18432
config.moe_intermediate_size = 2048
config.num_experts_per_tok = 8

# Mock the parallel info and parallel info manager
mock_parallel_info = MagicMock()
mock_parallel_info.rank = 0
mock_parallel_info.group_size = 1

mock_parallel_info_manager = MagicMock()
mock_parallel_info_manager.rank = 0
mock_parallel_info_manager.world_size = 1
mock_parallel_info_manager.attn_tp = mock_parallel_info
mock_parallel_info_manager.mlp_tp = mock_parallel_info
mock_parallel_info_manager.lm_head_tp = mock_parallel_info
mock_parallel_info_manager.moe_ep = mock_parallel_info
mock_parallel_info_manager.moe_tp = mock_parallel_info

# Set the global parallel info manager
set_parallel_info_manager(mock_parallel_info_manager)

quant_config = None


class TestDeepseekV3Moe(unittest.TestCase):

    @patch('torch.distributed.get_rank')
    def test_DeepseekV3Moe_init(self, mock_dist_get_rank):
        # Mock dist get rank to return local rank 0
        mock_dist_get_rank = MagicMock()
        mock_dist_get_rank.return_value = 0

        moe_prefix = "model.layers.3.mlp"
        moe_layer = DeepseekV3Moe(
            config=config,
            prefix=moe_prefix,
            quant_config=quant_config
        )

        self.assertIsInstance(moe_layer.experts, type(FusedMoE(
            num_experts=config.n_routed_experts,
            topk_num=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=f"{moe_prefix}.experts",
            suffix=["gate_proj", "down_proj", "up_proj"])))
        self.assertIsInstance(moe_layer.gate, type(ReplicatedLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            quant_config=quant_config,
            prefix=f"{moe_prefix}.gate",
        )))
        self.assertIsInstance(moe_layer.shared_experts, type(DeepseekV3MLP(
            config,
            f"{moe_prefix}.shared_experts",
            quant_config=quant_config,
            intermediate_size=config.moe_intermediate_size
        )))
        self.assertIsInstance(moe_layer.gate.e_score_correction_bias,
            type(BaseParameter(torch.empty(config.n_routed_experts))))

    @patch('mindie_llm.runtime.models.deepseek_v3.deepseek_v3.select_experts')
    @patch('mindie_llm.runtime.model_runner.forward_context.get_forward_context')
    @patch('torch.distributed.get_rank')
    def test_DeepseekV3Moe_forward(self, mock_dist_get_rank, mock_get_forward_context, mock_select):
        # Mock dist get rank to return local rank 0
        mock_dist_get_rank = MagicMock()
        mock_dist_get_rank.return_value = 0

        # Create inputs
        num_tokens = 10
        hidden_states = torch.randn(num_tokens, config.hidden_size)

        # Mock forward_context
        forward_context = MagicMock()
        forward_context.enable_flash_comm = False
        forward_context.seq_lens = torch.tensor([num_tokens])
        mock_get_forward_context.return_value = forward_context
        set_forward_context(forward_context)

        moe_prefix = "model.layers.3.mlp"
        moe_layer = DeepseekV3Moe(
            config=config,
            prefix=moe_prefix,
            quant_config=quant_config
        )
        fusedmoe = FusedMoE(
            num_experts=config.n_routed_experts,
            topk_num=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=f"{moe_prefix}.experts",
            suffix=["gate_proj", "down_proj", "up_proj"])
        fusedmoe.forward = MagicMock()
        fusedmoe.forward.return_value = torch.randn(num_tokens, config.hidden_size)
        moe_layer.experts = fusedmoe

        shared = DeepseekV3MLP(
            config,
            f"{moe_prefix}.shared_experts",
            quant_config=quant_config,
            intermediate_size=config.moe_intermediate_size
        )
        shared.forward = MagicMock()
        shared.forward.return_value = torch.randn(num_tokens, config.hidden_size)
        moe_layer.shared_experts = shared
        
        mock_select.return_value = (torch.randn(num_tokens, config.num_experts_per_tok), torch.randn(num_tokens, config.num_experts_per_tok))
     
        output = moe_layer(hidden_states)
        self.assertEqual(output.shape, (num_tokens, config.hidden_size))


class TestDeepseekV3Index(unittest.TestCase):

    def test_DeepseekV3Index_init(self):
        index_prefix = "model.layers.0.self_attn.indexer"
        deepseek_index = Indexer(
            config=config,
            prefix=index_prefix
        )

        self.assertIsInstance(deepseek_index.wq_b, type(ReplicatedLinear(
            config.q_lora_rank,
            config.index_n_heads * config.index_head_dim,
            bias=False,
            quant_config=None,
            prefix=f"{index_prefix}.wq_b"
        )))
        self.assertIsInstance(deepseek_index.wk,type(ReplicatedLinear(
            config.hidden_size,
            config.index_head_dim,
            bias=False,
            quant_config=None,
            prefix=f"{index_prefix}.wk"
        )))
        self.assertIsInstance(deepseek_index.weights_proj,type(ReplicatedLinear(
            config.hidden_size,
            config.index_head_dim,
            bias=False,
            quant_config=None, 
            prefix=f"{index_prefix}.weights_proj"
        )))
        self.assertIsInstance(deepseek_index.k_norm,type(LayerNorm(
            config.index_head_dim,
            config.rms_norm_eps,
            quant_config=None,
            prefix=f"{index_prefix}.k_norm"
        )))


class TestDeepseekV3Attn(unittest.TestCase):

    @patch('mindie_llm.runtime.layers.attention.backend.sparse_attention.SfaBackendImpl.check_parallel_info')
    def test_DeepseekV3Attn_init(self, mock_check):
        attn_prefix = "model.layers.0.self_attn"
        attn = DeepseekV3Attention(
            config=config,
            prefix=attn_prefix,
            quant_config=quant_config,
            enable_mlapo=False
        )

        self.assertIsInstance(attn.q_a_proj, type(ReplicatedLinear(
            config.hidden_size,
            config.q_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{attn_prefix}.q_a_proj"
        )))
        self.assertIsInstance(attn.kv_a_proj_with_mqa, type(ReplicatedLinear(
            config.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,                   
            bias=False,
            quant_config=quant_config,
            prefix=f"{attn_prefix}.kv_a_proj_with_mqa"
        )))
        self.assertIsInstance(attn.q_b_proj, type(ColumnParallelLinear(
            config.q_lora_rank,
            config.num_attention_heads * config.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{attn_prefix}.q_b_proj",
            parallel_info=mock_parallel_info_manager.attn_tp
        )))
        self.assertIsInstance(attn.kv_b_proj, type(ColumnParallelLinear(
            config.kv_lora_rank,
            config.num_attention_heads * (config.v_head_dim + config.qk_nope_head_dim),                  
            bias=False,
            quant_config=quant_config,
            prefix=f"{attn_prefix}.kv_b_proj",
            parallel_info=mock_parallel_info_manager.attn_tp
        )))
        self.assertIsInstance(attn.q_a_layernorm, type(RMSNorm(
            config.q_lora_rank,
            config.rms_norm_eps,
            quant_config=quant_config,
            prefix=f"{attn_prefix}.q_a_layernorm"
        )))
        self.assertIsInstance(attn.kv_a_layernorm, type(RMSNorm(
            config.kv_lora_rank,
            config.rms_norm_eps,
            quant_config=quant_config,
            prefix=f"{attn_prefix}.kv_a_layernorm"
        )))
        self.assertIsInstance(attn.o_proj, type(RowParallelLinear(
            config.num_attention_heads * config.qk_nope_head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{attn_prefix}.o_proj",
            parallel_info=mock_parallel_info_manager.attn_tp,
            reduce_results=True
        )))


class TestDeepseekV3MLP(unittest.TestCase):

    def test_DeepseekV3MLP_init(self):
        mlp_prefix = "model.layers.0.mlp"
        mlp = DeepseekV3MLP(
            config=config,
            prefix=mlp_prefix,
            quant_config=quant_config,
            intermediate_size=config.intermediate_size
        )
        self.assertIsInstance(mlp.gate_up_proj, type(MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=[f"{mlp_prefix}.gate_proj", f"{mlp_prefix}.up_proj"],
            parallel_info=mock_parallel_info_manager.mlp_tp
        )))
        self.assertIsInstance(mlp.down_proj, type(RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{mlp_prefix}.down_proj",
            parallel_info=mock_parallel_info_manager.mlp_tp,
            reduce_results=True
        )))

    @patch('mindie_llm.runtime.model_runner.forward_context.get_forward_context')
    @patch('mindie_llm.runtime.models.deepseek_v3.deepseek_v3.torch_npu.npu_swiglu')
    def test_DeepseekV3MLP_forward(self, mock_swiglu, mock_get_forward_context):
        mlp_prefix = "model.layers.0.mlp"
        mlp = DeepseekV3MLP(
            config=config,
            prefix=mlp_prefix,
            quant_config=quant_config,
            intermediate_size=config.intermediate_size
        )

        gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=[f"{mlp_prefix}.gate_proj", f"{mlp_prefix}.up_proj"],
            parallel_info=mock_parallel_info_manager.mlp_tp
        )
        down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{mlp_prefix}.down_proj",
            parallel_info=mock_parallel_info_manager.mlp_tp,
            reduce_results=False
        )
        gate_up_proj.forward = MagicMock()
        down_proj.forward = MagicMock()

        # Mock forward_context
        forward_context = MagicMock()
        forward_context.enable_flash_comm = False
        mock_get_forward_context.return_value = forward_context
        set_forward_context(forward_context)

        # Create inputs
        num_tokens = 10
        hidden_states = torch.randn(num_tokens, config.hidden_size)

        gate_up_proj.forward.return_value = torch.randn(num_tokens, 2 * config.hidden_size)
        mock_swiglu.return_value = torch.randn(num_tokens, 2 * config.hidden_size)
        down_proj.forward.return_value = torch.randn(num_tokens, config.hidden_size)

        mlp.gate_up_proj = gate_up_proj
        mlp.down_proj = down_proj

        output = mlp(hidden_states)
        self.assertEqual(output.shape, (num_tokens, config.hidden_size))


class TestDeepseekV3MlpLayer(unittest.TestCase):

    @patch('mindie_llm.runtime.layers.attention.backend.sparse_attention.SfaBackendImpl.check_parallel_info')
    @patch('mindie_llm.runtime.models.deepseek_v3.deepseek_v3.RMSNorm')
    @patch('mindie_llm.runtime.models.deepseek_v3.deepseek_v3.DeepseekV3Attention')
    @patch('mindie_llm.runtime.models.deepseek_v3.deepseek_v3.DeepseekV3MLP')    
    def test_DeepseekV3MlpLayer_init(self, mock_mlp, mock_attn, mock_rmsnorm, mock_check):
        layer_prefix = "model.layers.0"
        mlp_layer = DeepseekV3Layer(
            config=config,
            prefix=layer_prefix,
            layer_idx=0,
            quant_config=quant_config,
        )
        mock_mlp = MagicMock()
        mock_attn = MagicMock()
        mock_norm = MagicMock()
        self.assertIsInstance(mlp_layer.mlp, MagicMock)
        self.assertIsInstance(mlp_layer.self_attn, MagicMock)
        self.assertIsInstance(mlp_layer.input_layernorm, MagicMock)
        self.assertIsInstance(mlp_layer.post_attention_layernorm, MagicMock)
    
    @patch('mindie_llm.runtime.layers.attention.backend.sparse_attention.SfaBackendImpl.check_parallel_info')
    @patch('mindie_llm.runtime.model_runner.forward_context.get_forward_context')
    def test_DeepseekV3MlpLayer_forward(self, mock_get_forward_context, mock_check):
        prefix = "model.layers.0"
        mlp_layer = DeepseekV3Layer(
            config=config,
            prefix=prefix,
            layer_idx=0,
            quant_config=quant_config,
        )
        input_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            quant_config=quant_config,
            prefix=f"{prefix}.input_layernorm"
        )
        self_attn = DeepseekV3Attention(
            config, 
            f"{prefix}.self_attn", 
            quant_config=quant_config, 
            enable_mlapo=False,
            input_layernorm=None
        )
        post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            quant_config=quant_config,
            prefix=f"{prefix}.post_attention_layernorm"
        )
        mlp = DeepseekV3MLP(
            config=config,
            prefix=f"{prefix}.mlp",
            quant_config=quant_config,
            intermediate_size=config.intermediate_size
        )

        # Mock forward_context
        forward_context = MagicMock()
        forward_context.is_prefill = False
        mock_get_forward_context.return_value = forward_context
        set_forward_context(forward_context)

        input_layernorm.forward = MagicMock()
        self_attn.forward = MagicMock()
        post_attention_layernorm.forward = MagicMock()
        mlp.forward = MagicMock()

        num_tokens = 10
        hidden_states = torch.randn(num_tokens, config.hidden_size)
        input_layernorm.forward.return_value = torch.randn(num_tokens, config.hidden_size)
        self_attn.forward.return_value = torch.randn(num_tokens, config.hidden_size)
        post_attention_layernorm.forward.return_value = (torch.randn(num_tokens, config.hidden_size), torch.randn(num_tokens, config.hidden_size))
        mlp.forward.return_value = torch.randn(num_tokens, config.hidden_size)

        mlp_layer.input_layernorm = input_layernorm
        mlp_layer.self_attn = self_attn
        mlp_layer.post_attention_layernorm = post_attention_layernorm
        mlp_layer.mlp = mlp

        output = mlp_layer(hidden_states, None)
        self.assertEqual(output[0].shape, (num_tokens, config.hidden_size))


class TestDeepseekV3Model(unittest.TestCase):

    @patch('mindie_llm.runtime.layers.attention.backend.sparse_attention.SfaBackendImpl.check_parallel_info')
    @patch('mindie_llm.runtime.models.deepseek_v3.deepseek_v3.RMSNorm')
    @patch('torch.distributed.get_rank')
    def test_DeepseekV3Model_init(self, mock_dist_get_rank, mock_norm, mock_check):
        # Mock dist get rank to return local rank 0      
        mock_dist_get_rank = MagicMock()
        mock_dist_get_rank.return_value = 0

        model_prefix = "model"
        model = DeepseekV3Model(
            config=config,
            prefix=model_prefix,
            quant_config=quant_config
        )

        mock_norm = MagicMock()
        self.assertIsInstance(model.norm, MagicMock)
        self.assertIsInstance(model.embed_tokens, type(VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=None,
            prefix=f"{model_prefix}.embed_tokens",
        )))

    @patch('mindie_llm.runtime.layers.attention.backend.sparse_attention.SfaBackendImpl.check_parallel_info')
    @patch('mindie_llm.runtime.model_runner.forward_context.get_forward_context')
    @patch('torch.distributed.get_rank')
    def test_DeepseekV3Model_forward(self, mock_dist_get_rank, mock_get_forward_context, mock_check):
        # Mock dist get rank to return local rank 0
        mock_dist_get_rank = MagicMock()
        mock_dist_get_rank.return_value = 0

        model_prefix = "model"
        model = DeepseekV3Model(
            config=config,
            prefix=model_prefix,
            quant_config=quant_config
        )

        embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=None,
            prefix=f"{model_prefix}.embed_tokens",
        )
        layers = nn.ModuleList(
            [
                DeepseekV3Layer(config, model_prefix, layer_idx, quant_config)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        norm = RMSNorm(
            config.hidden_size, 
            config.rms_norm_eps, 
            quant_config=quant_config, 
            prefix=f"{model_prefix}.norm")
        
        embed_tokens.forward = MagicMock()
        for layer_idx in range(config.num_hidden_layers):
            layers[layer_idx].forward = MagicMock()
        norm.forward = MagicMock()

        # Create inputs
        num_tokens = 10
        input_ids = torch.randint(0, config.vocab_size, (num_tokens,))
        positions = torch.randint(0, 2048, (num_tokens,))

        # Mock forward_context
        forward_context = MagicMock()
        forward_context.seq_lens = torch.tensor([num_tokens])
        mock_get_forward_context.return_value = forward_context
        set_forward_context(forward_context)

        embed_tokens.forward.return_value = torch.randn(num_tokens, config.hidden_size)
        for layer_idx in range(config.num_hidden_layers):
            layers[layer_idx].forward.return_value = (
                torch.randn(num_tokens, config.hidden_size),
                torch.randn(num_tokens, config.hidden_size)
            )
        norm.forward.return_value = (
            torch.randn(num_tokens, config.hidden_size),
            torch.randn(num_tokens, config.hidden_size)
        )

        model.embed_tokens = embed_tokens
        model.layers = layers
        model.norm = norm

        output = model(input_ids, positions)
        self.assertEqual(output.shape, (num_tokens, config.hidden_size))


class TestDeepseekV3CausalLM(unittest.TestCase):

    @patch('mindie_llm.runtime.layers.attention.backend.sparse_attention.SfaBackendImpl.check_parallel_info')
    @patch('torch.distributed.get_rank')
    def test_DeepseekV3CausalLm_init(self, mock_dist_get_rank, mock_check):
        # Mock dist get rank to return local rank 0
        mock_dist_get_rank = MagicMock()
        mock_dist_get_rank.return_value = 0

        mindie_llm_config = MagicMock()
        mindie_llm_config.hf_config = config
        mindie_llm_config.quant_config = quant_config

        ds = DeepseekV3ForCausalLM(mindie_llm_config)

        expect_scale = (config.qk_nope_head_dim + config.qk_rope_head_dim) ** (-0.5)

        self.assertIsInstance(ds.lm_head, type(ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"lm_head",
        )))
        self.assertEqual(ds.kv_lora_rank, 512)
        self.assertEqual(ds.qk_rope_head_dim, 64)
        self.assertEqual(ds.softmax_scale, expect_scale)


if __name__ == '__main__':
    unittest.main()