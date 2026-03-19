# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from .plugin_manager import PluginManager
from .plugin_manager_lwd import PluginManagerLwd


def get_plugin(
    plugin_list,
    plugin_config,
    plugin_utils,
    is_mix_model,
    watcher
):
    generator_backend, kvcache_settings, infer_context, output_filter, model_role = plugin_utils
    if 'layerwise_disaggregated' in plugin_config and plugin_config['layerwise_disaggregated']:
        plugin_ins = PluginManagerLwd(
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
