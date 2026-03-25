# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import math
from typing import List

import torch


class FlashCommModifier:
    def __init__(self, weights, hidden_size, enable_flash_comm, **kwargs):
        self.hidden_size = hidden_size
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.active = False
        self.enable_flash_comm = enable_flash_comm
        
    @staticmethod
    def pass_flash_comm_threshold(input_ids_len, hidden_size) -> bool:
        threshold_comm_volume = 5242880 # 5242880 = 1024 * 5120, recommended threshold based on empirical testing
        actual_comm_volume = input_ids_len * hidden_size
        # Enable FlashComm when actual communication volume exceeds the threshold
        return actual_comm_volume > threshold_comm_volume

    def modify_inputs(self, inputs: List[torch.Tensor], is_prefill: bool, runtime_param) -> None:
        input_ids = inputs[0]
        # Check if FlashComm should be activated: requires feature enabled, prefill stage, 
        # and communication volume exceeding the threshold
        active_flag = all([
            self.enable_flash_comm,
            is_prefill,
            self.pass_flash_comm_threshold(input_ids.shape[0], self.hidden_size)
        ])
        if not active_flag:
            self.active = False
            return
        else:
            self.active = True
            # Flashcomm
            split_size = math.floor(input_ids.shape[0] / self.tp_world_size)
            remain_size = input_ids.shape[0] % self.tp_world_size
            # send_bs: input_ids list for each cards
            send_bs = torch.full((self.tp_world_size,), split_size, dtype=torch.int64)
            # send_counts: AllGatherV's and ReduceScatter's data, receive and send data for each cards
            send_counts = torch.full((self.tp_world_size,), split_size * self.hidden_size, dtype=torch.int64)
            for i in range(self.tp_world_size):
                if i < remain_size:
                    send_bs[i] += 1
                    send_counts[i] += 1 * self.hidden_size
            cum_counts = torch.cumsum(send_counts, dim=0)
            # sdispls: ReduceScatterV data, the offset list of the send input_ids for each card
            sdispls = torch.cat((torch.zeros(1, dtype=torch.int64), cum_counts[:-1]))
            cum_bs = torch.cumsum(send_bs, dim=0)
            # rdispls: AllGatherV data, the offset list of the received data for each card
            rdispls = torch.cat((torch.zeros(1, dtype=torch.int64), cum_bs[:-1]))
            send_count = torch.tensor([send_bs[self.tp_rank]], dtype=torch.int64)
            recv_count = torch.tensor([send_counts[self.tp_rank]], dtype=torch.int64)
            fake_rs_shape = torch.zeros([send_bs[self.tp_rank]], dtype=torch.float16)
            fake_ag_shape = torch.zeros([input_ids.shape[0]], dtype=torch.float16)
            runtime_param.update({
                "sendCounts": send_counts.tolist(),
                "sdispls": sdispls.tolist(),
                "sendCount": send_count.tolist(),
                "recvCounts": send_bs.tolist(),
                "rdispls": rdispls.tolist(),
                "recvCount": recv_count.tolist(),
            })
            inputs.append(send_counts.npu())
            inputs.append(sdispls.npu())
            inputs.append(send_count.npu())
            inputs.append(send_bs.npu())
            inputs.append(rdispls.npu())
            inputs.append(recv_count.npu())
            inputs.append(fake_rs_shape.npu())
            inputs.append(fake_ag_shape.npu())