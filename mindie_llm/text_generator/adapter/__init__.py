# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# redundant alias for ruff check

from ...modeling.backend_type import BackendType
from ...utils.log.error_code import ErrorCode
from ...utils.log.logging import logger


def get_generator_backend(model_config):
    backend_type = model_config.get('backend_type', None)
    if backend_type == BackendType.TORCH:
        from .generator_aclgraph import GeneratorAclGraph
        generator_cls = GeneratorAclGraph
    elif backend_type == BackendType.ATB:
        if model_config.get('async_inference', False):
            from .generator_torch_async import GeneratorTorchAsync
            generator_cls = GeneratorTorchAsync
        else:
            from .generator_torch import GeneratorTorch
            generator_cls = GeneratorTorch
    else:
        message = ('Unsupported backend type. The `backend_type` field only supports either "atb" or "ms". If you are '
                   'using a service framework, please modify its configuration file to ensure the `backend_type` '
                   'parameter passed to the `Generator` is correct. Such files are typically located in '
                   'conf/config.json or a similar path. Note that this parameter must be consistent with the '
                   'environment variable `MINDIE_LLM_FRAMEWORK_BACKEND`.')
        logger.error(message, ErrorCode.TEXT_GENERATOR_GENERATOR_BACKEND_INVALID)
        raise NotImplementedError(f'{backend_type} not implemented, '
                                  f'supported backends `{BackendType.__members__.values()}`')
    return generator_cls(model_config)