# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from mindie_llm.text_generator.adapter.generator_backend import GeneratorBackend
from mindie_llm.text_generator.utils.model_input import ModelInput
from mindie_llm.text_generator.utils.sampling_metadata import SamplingMetadata, SamplingData, SamplingParam
GENERATOR_BACKEND_AVAILABLE = True
_import_error = None

MOCKED_GET_MODEL_WRAPPER = "mindie_llm.text_generator.adapter.generator_backend.get_model_wrapper"
MOCKED_SAMPLER = "mindie_llm.text_generator.adapter.generator_backend.Sampler"


def get_default_model_config():
    """Return minimal model config for GeneratorBackend init."""
    return {
        'backend_type': 'atb',
        'npu_device_id': 0,
        'local_rank': 0,
        'rank': 0,
        'world_size': 1,
        'trust_remote_code': False,
    }


def create_mock_model_wrapper():
    """Create a mock model wrapper for tests."""
    mock_wrapper = MagicMock()
    mock_wrapper.config = MagicMock()
    mock_wrapper.config_dict = {'max_position_embeddings': 32768}
    mock_wrapper.model_info = None
    mock_wrapper.max_position_embeddings = 32768
    mock_wrapper.forward = MagicMock(return_value=torch.tensor([[0.1, 0.2, 0.7]]))
    mock_wrapper.make_context = MagicMock(return_value=[1, 2, 3])
    return mock_wrapper


@unittest.skipUnless(GENERATOR_BACKEND_AVAILABLE, f"torch_npu not installed: {_import_error or 'ok'}")
class TestGeneratorBackend(unittest.TestCase):
    """Unit tests for GeneratorBackend."""

    @patch(MOCKED_GET_MODEL_WRAPPER)
    def test_init_success(self, mock_get_wrapper):
        """Test successful initialization with valid config."""
        mock_wrapper = create_mock_model_wrapper()
        mock_get_wrapper.return_value = mock_wrapper

        config = get_default_model_config()
        backend = GeneratorBackend(config)

        self.assertEqual(backend.rank, 0)
        self.assertEqual(backend.world_size, 1)
        self.assertEqual(backend.npu_device_id, 0)
        self.assertEqual(backend.max_position_embeddings, 32768)
        mock_get_wrapper.assert_called_once()

    @patch(MOCKED_GET_MODEL_WRAPPER)
    def test_init_world_size_invalid_small(self, mock_get_wrapper):
        """Test init raises ValueError when world_size < 1."""
        mock_get_wrapper.return_value = create_mock_model_wrapper()
        config = get_default_model_config()
        config['world_size'] = 0

        with self.assertRaises(ValueError) as cm:
            GeneratorBackend(config)
        self.assertIn("World size should be in the range of 1 to 1048576", str(cm.exception))

    @patch(MOCKED_GET_MODEL_WRAPPER)
    def test_init_world_size_invalid_large(self, mock_get_wrapper):
        """Test init raises ValueError when world_size > MAX_WORLD_SIZE."""
        mock_get_wrapper.return_value = create_mock_model_wrapper()
        config = get_default_model_config()
        config['world_size'] = 1048577

        with self.assertRaises(ValueError) as cm:
            GeneratorBackend(config)
        self.assertIn("World size should be in the range of 1 to 1048576", str(cm.exception))

    @patch(MOCKED_GET_MODEL_WRAPPER)
    def test_init_rank_invalid_negative(self, mock_get_wrapper):
        """Test init raises ValueError when rank < 0."""
        mock_get_wrapper.return_value = create_mock_model_wrapper()
        config = get_default_model_config()
        config['rank'] = -1

        with self.assertRaises(ValueError) as cm:
            GeneratorBackend(config)
        self.assertIn("Rank should be in the range of 0 to world_size - 1", str(cm.exception))

    @patch(MOCKED_GET_MODEL_WRAPPER)
    def test_init_rank_invalid_exceeds_world_size(self, mock_get_wrapper):
        """Test init raises ValueError when rank >= world_size."""
        mock_get_wrapper.return_value = create_mock_model_wrapper()
        config = get_default_model_config()
        config['rank'] = 1
        config['world_size'] = 1

        with self.assertRaises(ValueError) as cm:
            GeneratorBackend(config)
        self.assertIn("Rank should be in the range of 0 to world_size - 1", str(cm.exception))

    @patch(MOCKED_GET_MODEL_WRAPPER)
    def test_init_local_rank_invalid(self, mock_get_wrapper):
        """Test init raises ValueError when local_rank is invalid."""
        mock_get_wrapper.return_value = create_mock_model_wrapper()
        config = get_default_model_config()
        config['local_rank'] = 2
        config['world_size'] = 1

        with self.assertRaises(ValueError) as cm:
            GeneratorBackend(config)
        self.assertIn("Local rank should be in the range of 0 to world_size - 1", str(cm.exception))

    def test_repeat_sample_param_none(self):
        """Test repeat_sample_param returns None when param_tensor is None."""
        result = GeneratorBackend.repeat_sample_param(None, [1, 2, 3])
        self.assertIsNone(result)

    def test_repeat_sample_param_valid(self):
        """Test repeat_sample_param with valid tensors."""
        param_tensor = [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]])]
        tokens_num_per_batch = [2, 1]
        result = GeneratorBackend.repeat_sample_param(param_tensor, tokens_num_per_batch)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[0], 3)  # 2 + 1

    @patch(MOCKED_SAMPLER)
    @patch(MOCKED_GET_MODEL_WRAPPER)
    def test_configure_sampler(self, mock_get_wrapper, mock_sampler_cls):
        """Test configure_sampler calls sampler.configure."""
        mock_get_wrapper.return_value = create_mock_model_wrapper()
        mock_sampler = MagicMock()
        mock_sampler_cls.return_value = mock_sampler
        backend = GeneratorBackend(get_default_model_config())
        sampling_metadata = MagicMock()

        backend.configure_sampler(sampling_metadata)
        mock_sampler.configure.assert_called_once_with(sampling_metadata)

    @patch(MOCKED_SAMPLER)
    @patch(MOCKED_GET_MODEL_WRAPPER)
    def test_init_sampler(self, mock_get_wrapper, mock_sampler_cls):
        """Test init_sampler calls sampler.initialize."""
        mock_get_wrapper.return_value = create_mock_model_wrapper()
        mock_sampler = MagicMock()
        mock_sampler_cls.return_value = mock_sampler
        backend = GeneratorBackend(get_default_model_config())
        backend.device = 'cpu'

        backend.init_sampler(2)  # eos_token_id
        mock_sampler.initialize.assert_called_once_with('cpu', 2)

    @patch(MOCKED_GET_MODEL_WRAPPER)
    def test_set_device(self, mock_get_wrapper):
        """Test set_device does nothing (pass)."""
        mock_get_wrapper.return_value = create_mock_model_wrapper()
        backend = GeneratorBackend(get_default_model_config())
        backend.set_device()  # Should not raise

    @patch(MOCKED_GET_MODEL_WRAPPER)
    def test_notify_force_stop_exception(self, mock_get_wrapper):
        """Test notify_force_stop_exception sets event."""
        mock_get_wrapper.return_value = create_mock_model_wrapper()
        backend = GeneratorBackend(get_default_model_config())

        self.assertFalse(backend.force_stop_exception_occurred.is_set())
        backend.notify_force_stop_exception()
        self.assertTrue(backend.force_stop_exception_occurred.is_set())

    @patch(MOCKED_GET_MODEL_WRAPPER)
    def test_execute_recover_command_unknown(self, mock_get_wrapper):
        """Test execute_recover_command with unknown command."""
        mock_get_wrapper.return_value = create_mock_model_wrapper()
        backend = GeneratorBackend(get_default_model_config())

        result = backend.execute_recover_command("UNKNOWN_CMD")
        self.assertEqual(result["command_result"], 1)
        self.assertEqual(result["npu_device_id"], 0)

    @patch(MOCKED_GET_MODEL_WRAPPER)
    def test_execute_recover_command_cmd_reinit_npu(self, mock_get_wrapper):
        """Test execute_recover_command with CMD_REINIT_NPU catches NotImplementedError."""
        mock_get_wrapper.return_value = create_mock_model_wrapper()
        backend = GeneratorBackend(get_default_model_config())

        result = backend.execute_recover_command("CMD_REINIT_NPU")
        self.assertEqual(result["command_result"], 1)
        self.assertIn("Subclasses must implement", result["error_msg"])

    @patch(MOCKED_GET_MODEL_WRAPPER)
    @patch("mindie_llm.text_generator.adapter.generator_backend.time.sleep")
    @patch("torch_npu.npu.stop_device")
    def test_execute_recover_command_cmd_pause_engine_stop_failed(
        self, mock_stop_device, mock_sleep, mock_get_wrapper
    ):
        """Test execute_recover_command when stop_device fails."""
        mock_get_wrapper.return_value = create_mock_model_wrapper()
        mock_stop_device.return_value = 1  # failure

        backend = GeneratorBackend(get_default_model_config())
        result = backend.execute_recover_command("CMD_PAUSE_ENGINE")

        self.assertEqual(result["command_result"], 1)
        self.assertIn("Stop device failed", result["error_msg"])

    @patch(MOCKED_GET_MODEL_WRAPPER)
    @patch("mindie_llm.text_generator.adapter.generator_backend.time.sleep")
    @patch("torch_npu.npu.stop_device")
    def test_execute_recover_command_cmd_pause_engine_uce_reschedule(
        self, mock_stop_device, mock_sleep, mock_get_wrapper
    ):
        """Test execute_recover_command when UCE requires reschedule."""
        mock_get_wrapper.return_value = create_mock_model_wrapper()
        mock_stop_device.return_value = 0

        backend = GeneratorBackend(get_default_model_config())
        backend._handle_uce_error = MagicMock(return_value=(1, "HBM uce address unknown"))

        result = backend.execute_recover_command("CMD_PAUSE_ENGINE")

        self.assertEqual(result["command_result"], 1)
        self.assertIn("HBM uce address unknown", result["error_msg"])

    @patch(MOCKED_GET_MODEL_WRAPPER)
    @patch("mindie_llm.text_generator.adapter.generator_backend.time.sleep")
    @patch("torch_npu.npu.stop_device")
    def test_execute_recover_command_cmd_pause_engine_uce_recovered(
        self, mock_stop_device, mock_sleep, mock_get_wrapper
    ):
        """Test execute_recover_command when UCE is recovered."""
        mock_get_wrapper.return_value = create_mock_model_wrapper()
        mock_stop_device.return_value = 0

        backend = GeneratorBackend(get_default_model_config())
        backend._handle_uce_error = MagicMock(return_value=(2, ""))

        result = backend.execute_recover_command("CMD_PAUSE_ENGINE")

        self.assertEqual(result["command_result"], 0)
        self.assertEqual(result["error_msg"], "")

    @patch(MOCKED_GET_MODEL_WRAPPER)
    @patch("mindie_llm.text_generator.adapter.generator_backend.time.sleep")
    @patch("torch_npu.npu.stop_device")
    def test_execute_recover_command_cmd_pause_engine_force_stop_timeout(
        self, mock_stop_device, mock_sleep, mock_get_wrapper
    ):
        """Test execute_recover_command when force stop times out."""
        mock_get_wrapper.return_value = create_mock_model_wrapper()
        mock_stop_device.return_value = 0

        backend = GeneratorBackend(get_default_model_config())
        backend._handle_uce_error = MagicMock(return_value=(0, ""))
        backend._wait_for_force_stop_exception = MagicMock(return_value=False)

        result = backend.execute_recover_command("CMD_PAUSE_ENGINE")

        self.assertEqual(result["command_result"], 1)
        self.assertIn("Timeout waiting for FORCE STOP exception", result["error_msg"])

    @patch(MOCKED_GET_MODEL_WRAPPER)
    @patch("mindie_llm.text_generator.adapter.generator_backend.time.sleep")
    @patch("torch_npu.npu.stop_device")
    def test_execute_recover_command_cmd_pause_engine_force_stop_success(
        self, mock_stop_device, mock_sleep, mock_get_wrapper
    ):
        """Test execute_recover_command when force stop succeeds."""
        mock_get_wrapper.return_value = create_mock_model_wrapper()
        mock_stop_device.return_value = 0

        backend = GeneratorBackend(get_default_model_config())
        backend._handle_uce_error = MagicMock(return_value=(0, ""))
        backend._wait_for_force_stop_exception = MagicMock(return_value=True)

        result = backend.execute_recover_command("CMD_PAUSE_ENGINE")

        self.assertEqual(result["command_result"], 0)
        self.assertEqual(result["error_msg"], "")

    @patch(MOCKED_GET_MODEL_WRAPPER)
    def test_build_inputs(self, mock_get_wrapper):
        """Test build_inputs calls make_context for each conversation."""
        mock_wrapper = create_mock_model_wrapper()
        mock_wrapper.make_context = MagicMock(side_effect=[[1, 2], [3, 4, 5]])
        mock_get_wrapper.return_value = mock_wrapper

        backend = GeneratorBackend(get_default_model_config())
        conversations = [[{"role": "user", "content": "hi"}], [{"role": "user", "content": "hello"}]]

        result = backend.build_inputs(conversations)

        self.assertEqual(result, [[1, 2], [3, 4, 5]])
        self.assertEqual(mock_wrapper.make_context.call_count, 2)

    @patch(MOCKED_SAMPLER)
    @patch(MOCKED_GET_MODEL_WRAPPER)
    def test_clear_cache(self, mock_get_wrapper, mock_sampler_cls):
        """Test clear_cache calls sampler.clear_cache."""
        mock_get_wrapper.return_value = create_mock_model_wrapper()
        mock_sampler = MagicMock()
        mock_sampler_cls.return_value = mock_sampler
        backend = GeneratorBackend(get_default_model_config())

        result = backend.clear_cache([1, 2, 3])
        mock_sampler.clear_cache.assert_called_once()
        self.assertEqual(result, 1)

    @patch(MOCKED_GET_MODEL_WRAPPER)
    def test_update_config(self, mock_get_wrapper):
        """Test update_config updates config attributes."""
        mock_wrapper = create_mock_model_wrapper()
        mock_wrapper.config_dict = {'max_position_embeddings': 32768}
        mock_get_wrapper.return_value = mock_wrapper

        backend = GeneratorBackend(get_default_model_config())
        backend.update_config({'max_position_embeddings': 8192})

        self.assertEqual(backend.config.max_position_embeddings, 8192)

    @patch(MOCKED_GET_MODEL_WRAPPER)
    def test_forward(self, mock_get_wrapper):
        """Test forward delegates to model_wrapper.forward."""
        mock_wrapper = create_mock_model_wrapper()
        expected_result = torch.tensor([[0.5, 0.3, 0.2]])
        mock_wrapper.forward = MagicMock(return_value=expected_result)
        mock_get_wrapper.return_value = mock_wrapper

        backend = GeneratorBackend(get_default_model_config())
        model_input = ModelInput(
            input_ids=np.array([1, 2, 3]),
            position_ids=np.array([0, 1, 2]),
            block_tables=np.array([[0]]),
            slots=np.array([0, 1, 2]),
            context_length=np.array([3]),
            max_seq_len=3,
            prefill_head_indices=np.array([2]),
            is_prefill=True,
            query_length=None,
            adapter_ids=None,
            dp_rank_ids=np.array([0]),
        )

        result = backend.forward(model_input)

        mock_wrapper.forward.assert_called_once_with(model_input)
        self.assertTrue(torch.equal(result, expected_result))

    @patch(MOCKED_GET_MODEL_WRAPPER)
    def test_sample_with_sampling_metadata(self, mock_get_wrapper):
        """Test sample with SamplingMetadata (non-deprecated path)."""
        mock_get_wrapper.return_value = create_mock_model_wrapper()
        backend = GeneratorBackend(get_default_model_config())

        logits = torch.tensor([[0.1, 0.2, 0.7]])
        sampling_metadata = SamplingMetadata.from_numpy(
            batch_sequence_ids=[np.array([0])],
            is_prefill=True,
            to_tensor=lambda x: torch.tensor(x) if x is not None else None,
        )

        mock_sampling_output = MagicMock()
        mock_sampling_output.token_ids = np.array([2])
        backend.sampler = MagicMock(return_value=mock_sampling_output)

        output = backend.sample(logits, sampling_metadata)
        backend.sampler.assert_called_once_with(logits, sampling_metadata)
        self.assertEqual(output, mock_sampling_output)


    @patch(MOCKED_GET_MODEL_WRAPPER)
    def test_execute_cmd_reinit_npu_raises(self, mock_get_wrapper):
        """Test _execute_cmd_reinit_npu raises NotImplementedError."""
        mock_get_wrapper.return_value = create_mock_model_wrapper()
        backend = GeneratorBackend(get_default_model_config())

        with self.assertRaises(NotImplementedError) as cm:
            backend._execute_cmd_reinit_npu()
        self.assertIn("Subclasses must implement", str(cm.exception))

    @patch(MOCKED_GET_MODEL_WRAPPER)
    def test_wait_for_force_stop_exception_is_fault_device(self, mock_get_wrapper):
        """Test _wait_for_force_stop_exception when is_fault_device is True."""
        mock_get_wrapper.return_value = create_mock_model_wrapper()
        backend = GeneratorBackend(get_default_model_config())
        backend.is_fault_device = True

        result = backend._wait_for_force_stop_exception()
        self.assertTrue(result)

    @patch(MOCKED_GET_MODEL_WRAPPER)
    def test_wait_for_force_stop_exception_detected(self, mock_get_wrapper):
        """Test _wait_for_force_stop_exception when event is set."""
        mock_get_wrapper.return_value = create_mock_model_wrapper()
        backend = GeneratorBackend(get_default_model_config())
        backend.force_stop_exception_occurred.set()

        result = backend._wait_for_force_stop_exception()
        self.assertTrue(result)

    @patch("torch.npu.check_uce_in_memory")
    @patch(MOCKED_GET_MODEL_WRAPPER)
    def test_handle_uce_error_no_uce(self, mock_get_wrapper, mock_check_uce):
        """Test _handle_uce_error when no UCE error (res=0)."""
        mock_get_wrapper.return_value = create_mock_model_wrapper()
        mock_check_uce.return_value = 0

        backend = GeneratorBackend(get_default_model_config())
        result, error_msg = backend._handle_uce_error()

        self.assertEqual(result, 0)
        self.assertEqual(error_msg, "")

    @patch("torch.npu.check_uce_in_memory")
    @patch(MOCKED_GET_MODEL_WRAPPER)
    def test_handle_uce_error_unknown_addr(self, mock_get_wrapper, mock_check_uce):
        """Test _handle_uce_error when UCE address unknown (res=1)."""
        mock_get_wrapper.return_value = create_mock_model_wrapper()
        mock_check_uce.return_value = 1

        backend = GeneratorBackend(get_default_model_config())
        result, error_msg = backend._handle_uce_error()

        self.assertEqual(result, 1)
        self.assertIn("uce address unknown", error_msg)

    @patch("torch.npu.check_uce_in_memory")
    @patch(MOCKED_GET_MODEL_WRAPPER)
    def test_handle_uce_error_not_in_kvcache(self, mock_get_wrapper, mock_check_uce):
        """Test _handle_uce_error when UCE not in kvcache (res=2 or 3)."""
        mock_get_wrapper.return_value = create_mock_model_wrapper()
        mock_check_uce.return_value = 2

        backend = GeneratorBackend(get_default_model_config())
        backend.cache_pool = MagicMock()
        backend.cache_pool.kvcache_settings = MagicMock()
        backend.cache_pool.kvcache_settings.num_layers = 1
        backend.cache_pool.npu_cache = [(torch.tensor([1, 2]), torch.tensor([3, 4]))]
        with patch("mindie_llm.text_generator.adapter.generator_backend.torch_npu.npu._get_uce_addr",
                  return_value=[{"ptr": 99999, "size": 100}]):
            backend._check_and_recover_uce_in_kvcache = MagicMock(return_value=False)

        result, error_msg = backend._handle_uce_error()

        self.assertEqual(result, 1)
        self.assertIn("not overlap kvcache address", error_msg)

    @patch("torch.npu.check_uce_in_memory")
    @patch(MOCKED_GET_MODEL_WRAPPER)
    def test_handle_uce_error_recovered(self, mock_get_wrapper, mock_check_uce):
        """Test _handle_uce_error when UCE is recovered."""
        mock_get_wrapper.return_value = create_mock_model_wrapper()
        mock_check_uce.return_value = 2

        backend = GeneratorBackend(get_default_model_config())
        backend.cache_pool = MagicMock()
        backend._check_and_recover_uce_in_kvcache = MagicMock(return_value=True)

        result, error_msg = backend._handle_uce_error()

        self.assertEqual(result, 2)
        self.assertEqual(error_msg, "")

    @patch("mindie_llm.text_generator.adapter.generator_backend.torch_npu.npu._get_uce_addr")
    @patch(MOCKED_GET_MODEL_WRAPPER)
    def test_check_and_recover_uce_in_kvcache_empty_list(self, mock_get_wrapper, mock_get_uce):
        """Test _check_and_recover_uce_in_kvcache with empty UCE list."""
        mock_get_wrapper.return_value = create_mock_model_wrapper()
        mock_get_uce.return_value = []

        backend = GeneratorBackend(get_default_model_config())
        backend.cache_pool = MagicMock()
        backend.cache_pool.kvcache_settings = MagicMock()
        backend.cache_pool.kvcache_settings.num_layers = 1
        backend.cache_pool.npu_cache = [(torch.tensor([1]), torch.tensor([2]))]

        result = backend._check_and_recover_uce_in_kvcache()
        self.assertFalse(result)

    @patch("mindie_llm.text_generator.adapter.generator_backend.check_and_recover_uce_in_cache")
    @patch("mindie_llm.text_generator.adapter.generator_backend.torch_npu.npu._get_uce_addr")
    @patch(MOCKED_GET_MODEL_WRAPPER)
    def test_check_and_recover_uce_in_kvcache_recovered(
        self, mock_get_wrapper, mock_get_uce, mock_check_recover
    ):
        """Test _check_and_recover_uce_in_kvcache when recovery succeeds."""
        mock_get_wrapper.return_value = create_mock_model_wrapper()
        mock_get_uce.return_value = [{"ptr": 100, "size": 50}]
        mock_check_recover.return_value = True

        backend = GeneratorBackend(get_default_model_config())
        backend.cache_pool = MagicMock()
        backend.cache_pool.kvcache_settings = MagicMock()
        backend.cache_pool.kvcache_settings.num_layers = 1
        backend.cache_pool.npu_cache = [(torch.tensor([1]), torch.tensor([2]))]

        result = backend._check_and_recover_uce_in_kvcache()
        self.assertTrue(result)
        mock_check_recover.assert_called()

    @patch("mindie_llm.text_generator.adapter.generator_backend.check_and_recover_uce_in_cache")
    @patch("mindie_llm.text_generator.adapter.generator_backend.torch_npu.npu._get_uce_addr")
    @patch(MOCKED_GET_MODEL_WRAPPER)
    def test_check_and_recover_uce_in_kvcache_not_recovered(
        self, mock_get_wrapper, mock_get_uce, mock_check_recover
    ):
        """Test _check_and_recover_uce_in_kvcache when recovery fails."""
        mock_get_wrapper.return_value = create_mock_model_wrapper()
        mock_get_uce.return_value = [{"ptr": 99999, "size": 100}]
        mock_check_recover.return_value = False

        backend = GeneratorBackend(get_default_model_config())
        backend.cache_pool = MagicMock()
        backend.cache_pool.kvcache_settings = MagicMock()
        backend.cache_pool.kvcache_settings.num_layers = 1
        backend.cache_pool.npu_cache = [(torch.tensor([1]), torch.tensor([2]))]

        result = backend._check_and_recover_uce_in_kvcache()
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
