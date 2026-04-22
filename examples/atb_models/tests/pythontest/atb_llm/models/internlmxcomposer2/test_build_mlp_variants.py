#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

import unittest
from types import SimpleNamespace

import torch
from unittest.mock import patch

from atb_llm.models.internlmxcomposer2.buildmlp import build_mlp
from atb_llm.models.internlmxcomposer2.buildmlp import build_mlp_4k


class TestBuildMlpVariants(unittest.TestCase):
    class _DummyVisionTower:
        def __init__(self, hidden_states, dtype=torch.float32, device="cpu"):
            self._hidden_states = hidden_states
            self.dtype = dtype
            self.device = torch.device(device)

        def __call__(self, inputs, output_hidden_states=True):
            return SimpleNamespace(hidden_states=self._hidden_states)

    def _mock_forward_outs(self):
        # hidden_states[-1] shape: [B, N, C]
        return SimpleNamespace(hidden_states=[torch.randn(1, 5, 8)])

    def test_build_mlp_feature_select_cls_patch(self):
        tower = build_mlp.CLIPVisionTower.__new__(build_mlp.CLIPVisionTower)
        tower.select_layer = -1
        tower.select_feature = "cls_patch"

        outs = self._mock_forward_outs()
        image_features = build_mlp.CLIPVisionTower.feature_select(tower, outs)

        self.assertEqual(image_features.shape, (1, 5, 8))

    def test_build_mlp_4k_feature_select_cls_patch(self):
        tower = build_mlp_4k.CLIPVisionTower.__new__(build_mlp_4k.CLIPVisionTower)
        tower.select_layer = -1
        tower.select_feature = "cls_patch"

        outs = self._mock_forward_outs()
        image_features = build_mlp_4k.CLIPVisionTower.feature_select(tower, outs)

        self.assertEqual(image_features.shape, (1, 5, 8))

    def test_build_mlp_feature_select_patch(self):
        tower = build_mlp.CLIPVisionTower.__new__(build_mlp.CLIPVisionTower)
        tower.select_layer = -1
        tower.select_feature = "patch"
        outs = self._mock_forward_outs()
        image_features = build_mlp.CLIPVisionTower.feature_select(tower, outs)
        self.assertEqual(image_features.shape, (1, 4, 8))

    def test_build_mlp_feature_select_invalid(self):
        tower = build_mlp.CLIPVisionTower.__new__(build_mlp.CLIPVisionTower)
        tower.select_layer = -1
        tower.select_feature = "invalid"
        with self.assertRaises(ValueError):
            build_mlp.CLIPVisionTower.feature_select(tower, self._mock_forward_outs())

    def test_build_mlp_4k_feature_select_invalid(self):
        tower = build_mlp_4k.CLIPVisionTower.__new__(build_mlp_4k.CLIPVisionTower)
        tower.select_layer = -1
        tower.select_feature = "invalid"
        with self.assertRaises(ValueError):
            build_mlp_4k.CLIPVisionTower.feature_select(tower, self._mock_forward_outs())

    def test_build_mlp_projector(self):
        projector = build_mlp.build_vision_projector()
        self.assertIsInstance(projector, torch.nn.Sequential)
        self.assertEqual(len(projector), 3)

    def test_build_mlp_4k_projector(self):
        projector = build_mlp_4k.build_vision_projector()
        self.assertIsInstance(projector, torch.nn.Sequential)
        self.assertEqual(len(projector), 3)

    def test_build_mlp_plora_forward(self):
        layer = build_mlp.PLoRA(4, 6, lora_dropout=0.0)
        x = torch.randn(2, 3, 4)
        out = layer(x)
        self.assertEqual(out.shape, (2, 3, 6))

    def test_build_mlp_plora_forward_with_all_true_mask(self):
        layer = build_mlp.PLoRA(4, 6, lora_dropout=0.0)
        x = torch.randn(1, 2, 4)
        mask = torch.ones(1, 2, dtype=torch.bool)
        out = layer(x, mask)
        self.assertEqual(out.shape, (1, 2, 6))

    def test_build_mlp_plora_forward_with_all_false_mask(self):
        layer = build_mlp.PLoRA(4, 6, lora_dropout=0.0)
        x = torch.randn(1, 2, 4)
        mask = torch.zeros(1, 2, dtype=torch.bool)
        out = layer(x, mask)
        self.assertEqual(out.shape, (1, 2, 6))

    def test_build_mlp_4k_plora_forward_mask(self):
        layer = build_mlp_4k.PLoRA(4, 6, lora_dropout=0.0)
        x = torch.randn(1, 2, 4)
        mask = torch.zeros(1, 2, dtype=torch.bool)
        out = layer(x, mask)
        self.assertEqual(out.shape, (1, 2, 6))

    def test_build_mlp_4k_forward_path(self):
        tower = build_mlp_4k.CLIPVisionTower.__new__(build_mlp_4k.CLIPVisionTower)
        tower.is_loaded = True
        tower.select_layer = -1
        tower.select_feature = "patch"
        tower.vision_tower = self._DummyVisionTower(
            hidden_states=[torch.randn(2, 577, 4)], dtype=torch.float32, device="cpu"
        )

        images = [torch.randn(1, 3, 336, 336)]
        glb_gn = torch.randn(1, 1, 16)
        sub_gn = torch.randn(1, 1, 1, 16)
        outputs, lens = build_mlp_4k.CLIPVisionTower.forward(tower, images, glb_gn, sub_gn)
        self.assertEqual(outputs.shape[0], 1)
        self.assertEqual(len(lens), 1)

    def test_build_mlp_forward_non_list_path(self):
        tower = build_mlp.CLIPVisionTower.__new__(build_mlp.CLIPVisionTower)
        tower.is_loaded = True
        tower.select_layer = -1
        tower.select_feature = "patch"
        tower.vision_tower = self._DummyVisionTower(
            hidden_states=[torch.randn(1, 5, 8)], dtype=torch.float32, device="cpu"
        )
        image = torch.randn(1, 3, 4, 4)
        features = build_mlp.CLIPVisionTower.forward(tower, image)
        self.assertEqual(features.shape, (1, 4, 8))

    def test_build_mlp_4k_load_model_path(self):
        tower = build_mlp_4k.CLIPVisionTower.__new__(build_mlp_4k.CLIPVisionTower)
        tower.is_loaded = False
        with patch.object(build_mlp_4k.CLIPVisionTower, "load_model", return_value=None):
            with self.assertRaises(AssertionError):
                build_mlp_4k.CLIPVisionTower.forward(tower, torch.randn(1, 3, 4, 4), None, None)

    def test_build_vision_tower_factory(self):
        fake_tower = object()
        with patch.object(build_mlp, "CLIPVisionTower", return_value=fake_tower):
            self.assertIs(build_mlp.build_vision_tower(), fake_tower)

    def test_build_vision_tower_factory_4k(self):
        fake_tower = object()
        with patch.object(build_mlp_4k, "CLIPVisionTower", return_value=fake_tower):
            self.assertIs(build_mlp_4k.build_vision_tower(), fake_tower)

    def test_identity_map_and_config(self):
        layer = build_mlp.IdentityMap()
        x = torch.randn(2, 3)
        self.assertTrue(torch.equal(layer(x), x))
        self.assertEqual(layer.config["mm_projector_type"], "identity")

    def test_identity_map_and_config_4k(self):
        layer = build_mlp_4k.IdentityMap()
        x = torch.randn(2, 3)
        self.assertTrue(torch.equal(layer(x), x))
        self.assertEqual(layer.config["mm_projector_type"], "identity")

    def test_clip_vision_tower_config_fallback(self):
        tower = build_mlp.CLIPVisionTower.__new__(build_mlp.CLIPVisionTower)
        tower.is_loaded = False
        tower.cfg_only = {"name": "cfg_only"}
        self.assertEqual(build_mlp.CLIPVisionTower.config.fget(tower), {"name": "cfg_only"})

    def test_clip_vision_tower_4k_config_fallback(self):
        tower = build_mlp_4k.CLIPVisionTower.__new__(build_mlp_4k.CLIPVisionTower)
        tower.is_loaded = False
        tower.cfg_only = {"name": "cfg_only"}
        self.assertEqual(build_mlp_4k.CLIPVisionTower.config.fget(tower), {"name": "cfg_only"})

    def test_clip_vision_tower_dummy_feature(self):
        tower = build_mlp.CLIPVisionTower.__new__(build_mlp.CLIPVisionTower)
        tower.is_loaded = True
        tower.vision_tower = SimpleNamespace(
            dtype=torch.float32,
            device=torch.device("cpu"),
            config=SimpleNamespace(hidden_size=6, image_size=336, patch_size=14),
        )
        feature = build_mlp.CLIPVisionTower.dummy_feature.fget(tower)
        self.assertEqual(feature.shape, (1, 6))

    def test_clip_vision_tower_4k_num_patches(self):
        tower = build_mlp_4k.CLIPVisionTower.__new__(build_mlp_4k.CLIPVisionTower)
        tower.is_loaded = True
        tower.vision_tower = SimpleNamespace(
            dtype=torch.float32,
            device=torch.device("cpu"),
            config=SimpleNamespace(hidden_size=6, image_size=336, patch_size=14),
        )
        self.assertEqual(build_mlp_4k.CLIPVisionTower.num_patches.fget(tower), 576)

    def test_build_vision_projector_unknown_type_raises(self):
        with patch.object(build_mlp.re, "match", return_value=None):
            with self.assertRaises(ValueError):
                build_mlp.build_vision_projector()

    def test_clip_vision_tower_resize_pos_when_already_target_size(self):
        tower = build_mlp.CLIPVisionTower.__new__(build_mlp.CLIPVisionTower)
        tower.is_resize_pos = False
        embed_dim = 8
        target_num_positions = 35**2 + 1
        pos_embed_weight = torch.zeros(target_num_positions, embed_dim)

        tower.vision_tower = SimpleNamespace(
            vision_model=SimpleNamespace(
                embeddings=SimpleNamespace(position_embedding=SimpleNamespace(weight=pos_embed_weight))
            )
        )

        build_mlp.CLIPVisionTower.resize_pos(tower)
        self.assertTrue(tower.is_resize_pos)

    def test_clip_vision_tower_forward_list_path(self):
        tower = build_mlp.CLIPVisionTower.__new__(build_mlp.CLIPVisionTower)
        tower.is_loaded = True
        tower.select_layer = -1
        tower.select_feature = "patch"
        tower.vision_tower = self._DummyVisionTower(
            hidden_states=[torch.randn(1, 5, 8)], dtype=torch.float32, device="cpu"
        )

        images = [torch.randn(3, 4, 4), torch.randn(3, 4, 4)]
        image_features = build_mlp.CLIPVisionTower.forward(tower, images)
        self.assertEqual(len(image_features), 2)
        self.assertEqual(image_features[0].shape, (1, 4, 8))

    def test_clip_vision_tower_init_invokes_load_and_resize(self):
        with (
            patch.object(build_mlp.CLIPVisionTower, "load_model") as m_load,
            patch.object(build_mlp.CLIPVisionTower, "resize_pos") as m_resize,
        ):
            build_mlp.CLIPVisionTower("openai/clip-vit-large-patch14-336")
        m_load.assert_called_once()
        m_resize.assert_called_once()

    def test_clip_vision_tower_load_model_sets_loaded(self):
        fake_vit = SimpleNamespace(requires_grad_=lambda _rg: None)
        tower = build_mlp.CLIPVisionTower.__new__(build_mlp.CLIPVisionTower)
        tower.vision_tower_name = "dummy-name"
        tower.is_loaded = False
        with patch.object(build_mlp, "safe_from_pretrained", return_value=fake_vit):
            build_mlp.CLIPVisionTower.load_model(tower)
        self.assertTrue(tower.is_loaded)
        self.assertIs(tower.vision_tower, fake_vit)

    def test_clip_vision_tower_resize_pos_interpolation(self):
        tower = build_mlp.CLIPVisionTower.__new__(build_mlp.CLIPVisionTower)
        tower.is_resize_pos = False
        n_tokens = 24 * 24 + 1
        embed_dim = 1024
        w = torch.randn(n_tokens, embed_dim)
        pos_emb = SimpleNamespace(weight=w)
        embeddings = SimpleNamespace(position_embedding=pos_emb)
        vision_model = SimpleNamespace(embeddings=embeddings)
        tower.vision_tower = SimpleNamespace(vision_model=vision_model)

        build_mlp.CLIPVisionTower.resize_pos(tower)

        self.assertTrue(tower.is_resize_pos)
        self.assertIsInstance(
            tower.vision_tower.vision_model.embeddings.position_embedding,
            torch.nn.Embedding,
        )

    def test_clip_vision_tower_forward_non_list_lazy_load(self):
        dummy = self._DummyVisionTower(hidden_states=[torch.randn(1, 5, 8)], dtype=torch.float32, device="cpu")
        tower = build_mlp.CLIPVisionTower.__new__(build_mlp.CLIPVisionTower)
        tower.is_loaded = False
        tower.select_layer = -1
        tower.select_feature = "patch"

        def _fake_load(*args, **_kwargs):
            model_self = args[0] if args else tower
            model_self.is_loaded = True
            model_self.vision_tower = dummy

        with patch.object(build_mlp.CLIPVisionTower, "load_model", side_effect=_fake_load):
            image = torch.randn(1, 3, 4, 4)
            features = build_mlp.CLIPVisionTower.forward(tower, image)
        self.assertEqual(features.shape, (1, 4, 8))

    def test_clip_vision_tower_forward_list_lazy_load(self):
        dummy = self._DummyVisionTower(hidden_states=[torch.randn(1, 5, 8)], dtype=torch.float32, device="cpu")
        tower = build_mlp.CLIPVisionTower.__new__(build_mlp.CLIPVisionTower)
        tower.is_loaded = False
        tower.select_layer = -1
        tower.select_feature = "patch"

        def _fake_load(*args, **_kwargs):
            model_self = args[0] if args else tower
            model_self.is_loaded = True
            model_self.vision_tower = dummy

        with patch.object(build_mlp.CLIPVisionTower, "load_model", side_effect=_fake_load):
            images = [torch.randn(3, 4, 4)]
            feats = build_mlp.CLIPVisionTower.forward(tower, images)
        self.assertEqual(len(feats), 1)
        self.assertEqual(feats[0].shape, (1, 4, 8))

    def test_build_mlp_plora_with_dropout(self):
        layer = build_mlp.PLoRA(4, 6, lora_dropout=0.1)
        self.assertIsInstance(layer.lora_dropout, torch.nn.Dropout)

    def test_build_mlp_plora_reset_parameters_lora_ab_branch(self):
        layer = build_mlp.PLoRA(4, 6, lora_dropout=0.0)
        layer.lora_A = torch.nn.Linear(4, 8, bias=False)
        layer.lora_B = torch.nn.Linear(8, 6, bias=False)
        build_mlp.PLoRA.reset_parameters(layer)


if __name__ == "__main__":
    unittest.main()
