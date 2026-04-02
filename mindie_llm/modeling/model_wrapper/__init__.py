#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from ..backend_type import BackendType
from ...utils.env import ENV


def get_model_wrapper(model_config, backend_type):
    if backend_type == BackendType.TORCH:
        if ENV.model_runner_exp:
            from .aclgraph.aclgraph_model_wrapper_exp import AclGraphModelWrapperExp
            wrapper_cls = AclGraphModelWrapperExp
        else:
            from .aclgraph.aclgraph_model_wrapper import AclGraphModelWrapper
            wrapper_cls = AclGraphModelWrapper
    elif backend_type == BackendType.ATB:
        from .atb.atb_model_wrapper import ATBModelWrapper
        wrapper_cls = ATBModelWrapper
    elif backend_type == BackendType.MS:
        from .ms.mf_model_wrapper import MFModelWrapper
        wrapper_cls = MFModelWrapper
    else:
        raise NotImplementedError(f'{backend_type} not implemented, '
                                  f'supported backends `{BackendType.__members__.values()}`')
    return wrapper_cls(**model_config)


def get_tokenizer_wrapper(model_id: str, backend_type: str, **kwargs):
    if backend_type == BackendType.TORCH:
        from .aclgraph.aclgraph_tokenizer_wrapper import AclGraphTokenizerWrapper
        wrapper_cls = AclGraphTokenizerWrapper
    elif backend_type == BackendType.ATB:
        from .atb.atb_tokenizer_wrapper import ATBTokenizerWrapper
        wrapper_cls = ATBTokenizerWrapper
    elif backend_type == BackendType.MS:
        from .ms.mf_tokenizer_wrapper import MFTokenizerWrapper
        wrapper_cls = MFTokenizerWrapper
    else:
        raise NotImplementedError(f'{backend_type} not implemented, '
                                  f'supported backends `{BackendType.__members__.values()}`')
    return wrapper_cls(model_id, **kwargs)
