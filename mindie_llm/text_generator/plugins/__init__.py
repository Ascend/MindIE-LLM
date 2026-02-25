# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import json
import re

from .plugin_manager import PluginManager
from .plugin_manager_edge import PluginManagerEdge
from .plugin_manager_lwd import PluginManagerLwd
from ...utils.log.error_code import ErrorCode
from ...utils.log.logging import logger, message_filter


PLUGIN_WHITE_LIST = ['la', 'memory_decoding', 'mtp', 'prefix_cache']
SPECULATIVE_PLUGIN_LIST = ['la', 'memory_decoding', 'mtp']
ASYNC_INFERENCE_UNSUPPORTED_OPTIONS = ["memory_decoding", "la"]


class InferenceMode:
    def __init__(self, enable_plugin_list, plugin_params, is_mix_model):
        self.enable_prefill_pa = False
        self.enable_decode_pa = False
        if is_mix_model or 'prefix_cache' in enable_plugin_list:
            self.enable_prefill_pa = True
        for plugin_type in enable_plugin_list:
            if plugin_type in SPECULATIVE_PLUGIN_LIST:
                self.enable_decode_pa = True
        self.enable_refactor = is_mix_model or len(enable_plugin_list) > 0
        if 'mtp' in enable_plugin_list:
            self.enable_refactor = False

    def __repr__(self):
        return (f"InferenceMode: \n"
                f"enable_prefill_pa = {self.enable_prefill_pa},"
                f"enable_decode_pa = {self.enable_decode_pa}")


PLUGIN_TYPE = 'plugin_type'


def get_plugin(
    plugin_list,
    plugin_config,
    plugin_utils,
    is_mix_model,
    watcher
):
    generator_backend, kvcache_settings, infer_context, output_filter, model_role = plugin_utils
    if 'soc_version' in plugin_config and plugin_config['soc_version'] == 'Ascend310B':
        plugin_ins = PluginManagerEdge(
            generator_backend,
            kvcache_settings,
            infer_context,
            output_filter,
            is_mix_model,
            plugin_list,
            **plugin_config
        )
    elif 'layerwise_disaggregated' in plugin_config and plugin_config['layerwise_disaggregated']:
        plugin_ins = PluginManagerLwd(generator_backend, kvcache_settings, infer_context, output_filter, is_mix_model,
            plugin_list, model_role, **plugin_config)
    else:
        plugin_ins = PluginManager(
            generator_backend,
            kvcache_settings,
            infer_context,
            output_filter,
            is_mix_model,
            plugin_list,
            model_role,
            watcher,
            **plugin_config
        )
    return plugin_ins



def validation_func_la(data, speculation_gamma):
    return (data.get('level') - 1) * (data.get('window') + data.get('guess_set_size')) <= speculation_gamma


def validation_func_mtp(data, speculation_gamma):
    num_speculative_tokens = data.get('num_speculative_tokens')
    max_reserved_len = max(num_speculative_tokens, num_speculative_tokens * 2 - 2)
    flag1 = max_reserved_len <= speculation_gamma
    flag2 = (num_speculative_tokens <= 5) and (num_speculative_tokens >= 0)
    return flag1 and flag2


PLUGIN_FIELDS = 'required_fields'
PLUGIN_CHECK_FUNC = 'validation_func'
MAX_JSON_LENGTH = 1024


class PluginParameterValidator:
    def __init__(self, speculation_gamma):
        self.speculation_gamma = speculation_gamma
        # 定义 plugin_type 和它所需要的字段之间的映射以及相应的校验函数
        # required_fields中添加必填参数，validation_func中添加对必填参数的校验逻辑
        self.rules = {
            'la': {
                PLUGIN_FIELDS: {'level', 'window', 'guess_set_size'},
                PLUGIN_CHECK_FUNC: lambda data: validation_func_la(data, self.speculation_gamma)
            },
            'memory_decoding': {
                PLUGIN_FIELDS: {'decoding_length'},
                PLUGIN_CHECK_FUNC: lambda data: data.get('decoding_length') <= self.speculation_gamma
            },
            'splitfuse': {
                PLUGIN_FIELDS: set(),  # 没有额外的字段要求
                PLUGIN_CHECK_FUNC: lambda data: True  # 不需要额外校验
            },
            'prefix_cache': {
                PLUGIN_FIELDS: set(),  # 没有额外的字段要求
                PLUGIN_CHECK_FUNC: lambda data: True  # 不需要额外校验
            },
            'mtp': {
                PLUGIN_FIELDS: {'num_speculative_tokens'},
                PLUGIN_CHECK_FUNC: lambda data: validation_func_mtp(data, self.speculation_gamma)
            }
        }
    
    @staticmethod
    def check_async_inference_and_plugin_type(async_inference: bool, plugin_types: str):
        if not async_inference or not plugin_types or not isinstance(plugin_types, str):
            return
        p_types = plugin_types.split(",")
        for tt in p_types:
            if tt in ASYNC_INFERENCE_UNSUPPORTED_OPTIONS:
                msg = f"The environment variable MINDIE_ASYNC_SCHEDULING_ENABLE does not support the plugin_type: {tt}."
                raise ValueError(msg)

    def validate(self, plugin_params):
        enabled_plugins_list = []
        is_mix_model = False
        if not plugin_params:
            plugin_config = {PLUGIN_TYPE: None}
            return plugin_config, is_mix_model, enabled_plugins_list

        # 验证给定的 JSON 字符串是否符合对应的 plugin_type 规则
        try:
            if len(plugin_params) > MAX_JSON_LENGTH:
                message = (f"The length of plugin_params is too long, it should be within (0, {MAX_JSON_LENGTH}]")
                raise ValueError(message)
            data = json.loads(plugin_params)
        except json.JSONDecodeError as e:
            message = ("The 'plugin_params' field does not conform to JSON format. Please check.")
            logger.error(message, ErrorCode.TEXT_GENERATOR_PLUGIN_NAME_INVALID)
            raise json.JSONDecodeError(message, plugin_params, e.pos) from e

        plugin_type_input = data.get(PLUGIN_TYPE)
        words = re.split(r',\s*', plugin_type_input)
        plugin_type_list = [word for word in words if word]
        for plugin_type in plugin_type_list:
            if plugin_type == "":
                plugin_config = {PLUGIN_TYPE: None}
                return plugin_config, is_mix_model, enabled_plugins_list

            rule = self.rules.get(plugin_type)
            if rule is None:
                message = (
                    f"Unsupported plugin type: {plugin_type}, "
                    f"Only 'la', 'memory_decoding', 'prefix_cache', 'mtp' and 'splitfuse' are supported.")
                message = message_filter(message)
                logger.error(message, ErrorCode.TEXT_GENERATOR_PLUGIN_NAME_INVALID)
                raise NotImplementedError(message)

            required_fields = rule[PLUGIN_FIELDS]
            missing_fields = required_fields.difference(data.keys())
            if missing_fields:
                message = (f"Missing fields for plugin_type '{plugin_type}': {missing_fields}")
                message = message_filter(message)
                logger.error(message, ErrorCode.TEXT_GENERATOR_PLUGIN_PARAM_VALUE_ERR)
                raise NotImplementedError(message)

            # 如果定义了校验函数，则执行校验
            if not rule[PLUGIN_CHECK_FUNC](data):
                message = (f"Validation failed for plugin_type '{plugin_type}'."
                           f" Please check the parameter configuration against the user manual.")
                message = message_filter(message)
                logger.error(message, ErrorCode.TEXT_GENERATOR_PLUGIN_PARAM_VALUE_ERR)
                raise ValueError(message)

            if plugin_type == 'splitfuse':
                is_mix_model = True

        for plugin in PLUGIN_WHITE_LIST:
            if plugin in plugin_type_list:
                enabled_plugins_list.append(plugin)

        return data, is_mix_model, enabled_plugins_list