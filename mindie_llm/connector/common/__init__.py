# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
def send_model_execute_response(proto, is_binary=False):
    from mindie_llm.connector.request_listener.shared_mem_communication import SharedMemCommunication
    if is_binary:
        SharedMemCommunication.send_model_execute_binary_data_cls(proto)
    else:
        SharedMemCommunication.send_model_execute_response_cls(proto)


def send_transfer_response(proto):
    from mindie_llm.connector.request_listener.shared_mem_communication import SharedMemCommunication
    SharedMemCommunication.send_transfer_response_cls(proto)


def send_command_response(proto):
    from mindie_llm.connector.request_listener.shared_mem_communication import SharedMemCommunication
    SharedMemCommunication.send_command_response_cls(proto)
