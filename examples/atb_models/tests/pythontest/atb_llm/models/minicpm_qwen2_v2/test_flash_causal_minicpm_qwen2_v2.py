#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

import unittest
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image
import numpy as np
import torch


def _import_target_module():
    # `flash_causal_minicpm_qwen2_v2` imports `.resampler`, but that module is absent
    # in some CI environments. Inject a lightweight stub before importing test target.
    resampler_module_name = "atb_llm.models.minicpm_qwen2_v2.resampler"
    if resampler_module_name not in sys.modules:
        fake_resampler = types.ModuleType(resampler_module_name)

        class _FakeResampler:  # pragma: no cover
            pass

        fake_resampler.Resampler = _FakeResampler
        sys.modules[resampler_module_name] = fake_resampler

    navit_module_name = "atb_llm.models.minicpm_qwen2_v2.modeling_navit_siglip"
    if navit_module_name not in sys.modules:
        fake_navit = types.ModuleType(navit_module_name)

        class _FakeSiglipVisionTransformer:  # pragma: no cover
            pass

        fake_navit.SiglipVisionTransformer = _FakeSiglipVisionTransformer
        sys.modules[navit_module_name] = fake_navit

    input_builder_module_name = "atb_llm.models.minicpm_qwen2_v2.input_builder_minicpm_qwen2_v2"
    if input_builder_module_name not in sys.modules:
        fake_input_builder = types.ModuleType(input_builder_module_name)

        def _fake_encode_video(*args, **kwargs):  # pragma: no cover
            return []

        fake_input_builder.encode_video = _fake_encode_video
        sys.modules[input_builder_module_name] = fake_input_builder

    from atb_llm.models.minicpm_qwen2_v2 import flash_causal_minicpm_qwen2_v2 as target

    return target


def _fake_multimodal_init(self, config, weights, **kwargs):
    self.soc_info = SimpleNamespace(need_nz=False)


def _fake_init_multimodal(self):
    self.vision_tower = SimpleNamespace(embed_dim=64)
    self.language_model = MagicMock()


class TestFlashCausalMiniCpmQwen2V2(unittest.TestCase):
    def test_init_calls_parent_and_sets_basic_fields(self):
        target = _import_target_module()
        config = SimpleNamespace(
            quantize=None,
            vocab_size=1000,
            hidden_size=128,
            num_attention_heads=8,
            num_hidden_layers=2,
            num_key_value_heads=2,
            pad_token_id=0,
            model_type="minicpm_qwen2_v2",
            vision_config=SimpleNamespace(),
            text_config=SimpleNamespace(),
        )
        weights = MagicMock()

        with (
            patch.object(target.FlashMinicpmqwen2v2ForCausalLM, "init_resampler", return_value=None),
            patch.object(
                target.FlashMinicpmqwen2v2ForCausalLM,
                "init_multimodal",
                side_effect=_fake_init_multimodal,
                autospec=True,
            ),
            patch.object(
                target.MultiModalLLm,
                "__init__",
                side_effect=_fake_multimodal_init,
                autospec=True,
            ) as mock_parent_init,
            patch(
                "atb_llm.models.base.flash_causal_lm.FlashForCausalLM.__init__",
                return_value=None,
            ),
        ):
            model = target.FlashMinicpmqwen2v2ForCausalLM(config, weights)

        mock_parent_init.assert_called_once()
        self.assertEqual(model.image_token_id, target.IMAGE_PAD)
        self.assertEqual(model.vision_start_token_id, target.VISION_START_TOKEN_ID)
        self.assertEqual(model.vision_end_token_id, target.VISION_END_TOKEN_ID)

    def test_process_qs_with_image_and_text(self):
        target = _import_target_module()
        text = [{"role": "user", "content": "describe this image"}]
        image = Image.new("RGB", (2, 2), color="white")

        new_text, images = target.process_qs(text, image)

        self.assertEqual(len(images), 1)
        self.assertEqual(new_text[0]["content"], "(<image>./</image>)\ndescribe this image")

    def test_process_qs_empty_text_raises(self):
        target = _import_target_module()
        with self.assertRaises(RuntimeError):
            target.process_qs([], None)

    def test_process_qs_invalid_role_raises(self):
        target = _import_target_module()
        with self.assertRaises(RuntimeError):
            target.process_qs([{"role": "system", "content": "x"}], None)

    def test_process_qs_first_role_not_user_raises(self):
        target = _import_target_module()
        with self.assertRaises(RuntimeError):
            target.process_qs([{"role": "assistant", "content": "x"}], None)

    def test_process_qs_string_input_json(self):
        target = _import_target_module()
        text = '[{"role":"user","content":"hello"}]'
        new_text, images = target.process_qs(text, None)
        self.assertEqual(images, [])
        self.assertEqual(new_text[0]["content"], "hello")

    def test_init_resamplerweight_sets_nested_parameter(self):
        target = _import_target_module()

        class FakeModule:
            def __init__(self):
                self.layer = SimpleNamespace(weight=None)

            def state_dict(self):
                return {"layer.weight": torch.zeros(1)}

        weights = MagicMock()
        weights.get_tensor.return_value = torch.ones(1)
        module = FakeModule()

        target.FlashMinicpmqwen2v2ForCausalLM.init_resamplerweight(module, weights)
        self.assertTrue(hasattr(module.layer, "weight"))
        self.assertEqual(float(module.layer.weight.data[0]), 1.0)

    def test_process_qs_ignores_unsupported_content_type(self):
        target = _import_target_module()
        text = [{"role": "user", "content": ["hello", 123]}]
        new_text, images = target.process_qs(text, None)
        self.assertEqual(images, [])
        self.assertEqual(new_text[0]["content"], "hello")

    def test_prepare_vision_embeds_without_pixel_values(self):
        target = _import_target_module()
        model = target.FlashMinicpmqwen2v2ForCausalLM.__new__(target.FlashMinicpmqwen2v2ForCausalLM)

        def _fake_embedding(input_ids):
            return torch.zeros(input_ids.shape[0], input_ids.shape[1], 4, dtype=torch.float32)

        model.get_input_embeddings = lambda: _fake_embedding
        model.image_token_id = target.IMAGE_PAD
        model.device = torch.device("cpu")
        model.dtype = torch.float32

        input_ids = torch.tensor([[1, 2]], dtype=torch.int64)
        out = target.FlashMinicpmqwen2v2ForCausalLM.prepare_vision_embeds(model, input_ids, [], [])
        self.assertEqual(out.shape, (1, 2, 4))

    def test_prepare_prefill_token_video_branch(self):
        target = _import_target_module()
        model = target.FlashMinicpmqwen2v2ForCausalLM.__new__(target.FlashMinicpmqwen2v2ForCausalLM)
        model.device = torch.device("cpu")

        class _FakeInputs(dict):
            def to(self, device):
                return self

        fake_inputs = _FakeInputs(
            {
                target.INPUT_IDS: torch.tensor([[1, 2]], dtype=torch.int64),
                target.PIXEL_VALUES: [[torch.zeros(3, 2, 2)]],
                target.TGT_SIZES: [torch.tensor([1, 1], dtype=torch.int32)],
            }
        )

        processor = MagicMock()
        processor.tokenizer.apply_chat_template.return_value = "prompt"
        processor.return_value = fake_inputs
        model.prepare_vision_embeds = MagicMock(return_value=torch.zeros(2, 4, dtype=torch.float32).unsqueeze(0))

        mm_inputs = SimpleNamespace(text={target.MSG_CONTENT: "hi"}, image=None, video="fake.mp4")
        with patch.object(target, "encode_video", return_value=["f1", "f2"]):
            result = target.FlashMinicpmqwen2v2ForCausalLM.prepare_prefill_token(model, mm_inputs, processor)
        self.assertEqual(len(result), 1)

    def test_prepare_vision_embeds_with_pixel_values(self):
        target = _import_target_module()

        batch_size = 2
        embed_dim = 4

        model = target.FlashMinicpmqwen2v2ForCausalLM.__new__(target.FlashMinicpmqwen2v2ForCausalLM)
        model.image_token_id = target.IMAGE_PAD
        model.device = torch.device("cpu")
        model.dtype = torch.float32
        # 保证走 prepare_vision_embeds() 的 batch_size <= max_vision_bs 分支
        model.max_vision_bs = 1024

        # mock LLM token embeddings：返回固定维度的 inputs_embeds
        model.get_input_embeddings = lambda: (  # noqa: E731
            lambda ids: torch.zeros(ids.shape[0], ids.shape[1], embed_dim, dtype=torch.float32)
        )

        def _fake_vision_tower(image_pixel_array, patch_attention_mask=None, tgt_sizes=None):
            # 生成形状为 [B, 1, C] 的 last_hidden_state，便于与 image_mask 替换对齐
            return SimpleNamespace(last_hidden_state=torch.ones(batch_size, 1, embed_dim, dtype=torch.float32))

        model.vision_tower = _fake_vision_tower
        model.resampler = lambda image_features, tgt_sizes: image_features  # noqa: E731

        input_ids = torch.full((batch_size, 1), model.image_token_id, dtype=torch.int64)
        pixel_values = [torch.zeros(3, 2, 2, dtype=torch.float32) for _ in range(batch_size)]
        tgt_sizes = [torch.tensor([1, 1], dtype=torch.int32) for _ in range(batch_size)]

        out = target.FlashMinicpmqwen2v2ForCausalLM.prepare_vision_embeds(model, input_ids, pixel_values, tgt_sizes)
        self.assertEqual(out.shape, (batch_size, 1, embed_dim))
        self.assertTrue(torch.all(out == 1.0))

    def test_prepare_prefill_token_service_reads_shm_and_replaces_special_tokens(self):
        target = _import_target_module()
        model = target.FlashMinicpmqwen2v2ForCausalLM.__new__(target.FlashMinicpmqwen2v2ForCausalLM)
        model.device = torch.device("cpu")
        model.vision_start_token_id = target.VISION_START_TOKEN_ID
        model.vision_end_token_id = target.VISION_END_TOKEN_ID
        model.image_token_id = target.IMAGE_PAD

        input_ids = torch.tensor(
            [
                target.VISION_START_TOKEN_ID,
                -10,
                -11,
                -12,
                -13,
                target.VISION_END_TOKEN_ID,
            ],
            dtype=torch.int64,
        )

        def _fake_get_data_from_shm(shm_name, shape_value, dtype, device=None):
            if dtype is np.int64:
                return torch.tensor([1, 1], dtype=torch.int64)
            return torch.zeros(3, 2, 2, dtype=torch.float32)

        with (
            patch.object(target, "get_data_from_shm", side_effect=_fake_get_data_from_shm),
            patch.object(
                target.FlashMinicpmqwen2v2ForCausalLM,
                "prepare_vision_embeds",
                return_value=torch.zeros(3, 4, dtype=torch.float32),
            ) as mock_prepare_vision_embeds,
        ):
            output = target.FlashMinicpmqwen2v2ForCausalLM.prepare_prefill_token_service(model, input_ids)

        self.assertEqual(output.shape, (3, 4))
        # prepare_vision_embeds(self, input_ids, pixel_values, tgt_sizes) — first arg after self is input_ids
        called_input_ids = mock_prepare_vision_embeds.call_args[0][0]
        self.assertTrue(torch.all(called_input_ids == target.IMAGE_PAD))

    def test_prepare_prefill_token_image_branch(self):
        target = _import_target_module()
        model = target.FlashMinicpmqwen2v2ForCausalLM.__new__(target.FlashMinicpmqwen2v2ForCausalLM)
        model.device = torch.device("cpu")

        class _FakeInputs(dict):
            def to(self, device):
                return self

        fake_inputs = _FakeInputs(
            {
                target.INPUT_IDS: torch.tensor([[1, 2]], dtype=torch.int64),
                target.PIXEL_VALUES: [[torch.zeros(3, 2, 2)]],
                target.TGT_SIZES: [torch.tensor([1, 1], dtype=torch.int32)],
            }
        )

        processor = MagicMock()
        processor.tokenizer.apply_chat_template.return_value = "prompt"
        processor.return_value = fake_inputs
        model.prepare_vision_embeds = MagicMock(return_value=torch.zeros(2, 4, dtype=torch.float32).unsqueeze(0))

        mm_inputs = SimpleNamespace(text={target.MSG_CONTENT: "hi"}, image="fake.jpg", video=None)
        with (
            patch.object(target, "safe_open_image", return_value=Image.new("RGB", (2, 2))),
            patch.object(
                target,
                "process_qs",
                return_value=([{"role": "user", target.MSG_CONTENT: "hello"}], [Image.new("RGB", (2, 2))]),
            ),
        ):
            result = target.FlashMinicpmqwen2v2ForCausalLM.prepare_prefill_token(model, mm_inputs, processor)

        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
