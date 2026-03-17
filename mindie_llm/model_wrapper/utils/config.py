#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import subprocess
import re
from pathlib import Path
import numpy as np
from dataclasses import dataclass, fields
from mindie_llm.text_generator.utils.separate_deployment_engine import DmiModeNodeRole
from mindie_llm.utils.log.logging import logger
from mindie_llm.model_wrapper.utils.common_util import ip_array_to_string, generate_dp_inst_id

SERVER_LIST = "server_list"
ATTRIBUTES_IDX = 0
DEVICES_IDX = 1
POLICY_IDX = 2


@dataclass
class LinkMapParams:
    role: DmiModeNodeRole
    tp_p: int = 1
    tp_d: int = 1
    sp_p: int = 1
    sp_d: int = 1
    cp_p: int = 1
    cp_d: int = 1

    def __post_init__(self):
        for f in fields(self):
            if f.name == 'role':
                continue
            value = getattr(self, f.name)
            if isinstance(value, int) and value < 1:
                raise ValueError(f"Field '{f.name}' must be a positive integer (>=1).")
            

class BaseConfig():
    def __init__(self, model_config: dict):
        if model_config is None:
            raise ValueError("model_config is None!")
        logger.info("model_config %s", model_config)

        self.model_config = model_config
        self.local_rank = self.parse('local_rank', required=True, to_int=True)
        self.infer_mode = self.parse('infer_mode', required=False, default_value="standard")

        self.remote_super_pod_id = {}
        self.remote_super_device_id = {}

        self.initialize()
        self.layerwise_disaggregated_initialize()

        logger.info(">>model_config after initialize %s", model_config)

    @staticmethod
    def check_path_is_mounted(target_path: str) -> bool:
        # 根据系统补充
        shared_fs_types = ["nfs", "cifs", "smbfs", "glusterfs", "lustre", "panfs", "afs", "ceph", "dtfs"]
        target_path = Path(target_path).resolve()
        mount_output = subprocess.check_output(["/usr/bin/mount"], encoding="utf-8", stderr=subprocess.STDOUT)
        mount_pattern = re.compile(r"^(.+?) on (.+?) type (.+?) \((.+?)\)$", re.MULTILINE)
        for match in mount_pattern.finditer(mount_output):
            _, mount_point, fs_type, _ = match.groups()
            mount_point = Path(mount_point).resolve()
            if (target_path == mount_point or target_path.is_relative_to(mount_point)) and fs_type in shared_fs_types:
                return True
        return False
    
    def layerwise_disaggregated_initialize(self):
        self.layerwise_disaggregated = self.parse("layerwiseDisaggregated", required=False, default_value=None)
        self.p_inst_enable_sp_cp = False    # 分布式边云协同场景暂不支持

    def initialize(self):
        # optional
        self.model_instance_type = self.parse("model_instance_type")
        self.speculation_gamma = self.parse("speculation_gamma")
        self.backend_type = self.parse('backend_type', default_value="")
        self.plugin_params = self.parse("plugin_params", default_value="")
        self.tp_size = self.parse("tp", default_value=1, to_int=True)
        self.pp_size = self.parse("pp", default_value=1)
        self.ep_size = self.parse("ep", default_value=1)
        self.dp_size = self.parse("dp", default_value=1, to_int=True)
        self.es_size = self.parse("es", default_value=1)
        self.sp_size = self.parse("sp", default_value=1, to_int=True)
        self.cp_size = self.parse("cp", default_value=1, to_int=True)
        # not optional
        self.model_weight_path = self.parse('model_id', required=True)

        self.trust_remote_code = self.parse('trust_remote_code', default_value=False)
        self.local_rank = self.parse("local_rank", required=True, to_int=True)
        self.rank = self.parse('rank', required=True, to_int=True)
        self.global_rank = self.parse('globalRankIds')
        if self.global_rank is not None:
            self.global_rank = self.global_rank.split(",")
            if self.global_rank != ['']:
                self.rank = int(self.global_rank[self.local_rank])

        self.global_world_size = self.parse("globalWorldSize", to_int=True)
        self.world_size = self.parse('world_size', required=True, to_int=True)
        self.npu_device_id = self.parse('npu_device_id', required=True, to_int=True)
        self.npu_device_ids = self.parse_list('npu_device_ids', required=True, to_int=True)
        self.cpu_mem_size = self.parse('cpu_mem', required=True, to_int=True)
        self.npu_mem_size = self.parse('npu_mem', required=True, to_int=True)
        self.max_seq_len = self.parse('max_seq_len', required=True, to_int=True)
        self.max_iter_times = self.parse('max_iter_times', required=True, to_int=True)
        self.max_prefill_tokens = self.parse('max_prefill_tokens', required=True, to_int=True)
        self.cache_block_size = self.parse("block_size", required=True, to_int=True)

        self.distributed_enable = self.parse("distributed_enable", required=True) == 'true'

    def parse(self, item_name, required=False, to_int=False, default_value=None):
        value = self.model_config.get(item_name)
        if value is None:
            if required:
                raise ValueError(f"You should set {item_name}.")
            if default_value is not None:
                logger.info(f"There is no item named {item_name}, use default value {default_value}.")
                value = default_value
        elif to_int:
            value = int(value)
        if item_name == "model_id":
            if BaseConfig.check_path_is_mounted(value):
                error_msg = f"The model {value} resides in mounted directory, which could be a shared storage path. \
                              If the I/O rate is too low, it may cause abnormal service startup."
                logger.warning(error_msg)
        logger.info(f"The item {item_name} value is {value}.")
        return value

    def parse_list(self, item_name, split_char=',', required=False, to_int=False, default_value=None):
        value = self.model_config.get(item_name)
        if value is None:
            if required:
                raise ValueError(f"You should set {item_name}.")
            if default_value is not None:
                logger.info(f"There is no item named {item_name}, use default value {default_value}.")
                value = default_value
        str_list = value.split(split_char)
        logger.info(f"The list item {item_name} value is {str_list}.")
        if to_int:
            int_list = [int(x) for x in str_list]
            return int_list
        return str_list


class DmiConfig(BaseConfig):
    def __init__(self, model_config: dict):
        super().__init__(model_config)

        self.role = "unknown"
        self.link_map = dict()
        # link and unlink info
        self.p_inst_enable_sp_cp = False
        self.remote_sp_size = 0
        self.remote_cp_size = 0

        self.remote_link_cluster_id = {}
        self.remote_link_host_ip = {}
        self.remote_link_device_ips = {}
        self.remote_link_device_physical_id = {}

        self.remote_unlink_cluster_id = {}
        self.remote_unlink_device_ips = {}
        self.need_switch = False
        self.local_dp_rank_to_id = list()
        self.dp_inst_id_to_cluster_id = {}

        self.init_dmi_config()

    @staticmethod
    def generate_link_map(params: LinkMapParams):
        d_to_p = {}
        
        # 根据 sp_p 的值选择不同的映射逻辑
        if params.sp_p != 1:
            # SP 模式的映射逻辑
            if params.sp_p <= 0 or params.tp_p % params.sp_p != 0:
                raise ValueError(
                    f"Invalid tp mapping (sp): tp_p ({params.tp_p}) must be divisible by sp_p ({params.sp_p})."
                )
            for rank_d in range(params.tp_d):
                d_to_p[rank_d] = [i * params.tp_p // params.sp_p for i in range(params.sp_p * params.cp_p)]
        else:
            # 标准模式的映射逻辑
            if params.tp_d == 0 or params.tp_p == 0:
                raise ValueError(
                    f"Invalid tp mapping: tp_p ({params.tp_p}) and tp_d ({params.tp_d}) must not equal 0")
            if params.tp_p % params.tp_d != 0 and params.tp_d % params.tp_p != 0:
                raise ValueError(
                    f"Invalid tp mapping: tp_p ({params.tp_p}) must be divisible by tp_d ({params.tp_d}) or "
                    f"tp_d ({params.tp_d}) must be divisible by tp_p ({params.tp_p})"
                )
            factor = params.tp_p // params.tp_d if params.tp_p >= params.tp_d else params.tp_d // params.tp_p
            for rank_d in range(params.tp_d):
                rank_d_base = rank_d // factor if params.tp_d >= params.tp_p else rank_d * factor
                d_to_p[rank_d] = [rank_d_base + i for i in range(0, params.cp_p * params.tp_p, params.tp_p)]

        # 如果是 DECODER 角色，直接返回 d_to_p
        if params.role == DmiModeNodeRole.DECODER:
            return d_to_p

        # 否则构建并返回 p_to_d
        p_to_d = {}
        for rank_d, rank_p in d_to_p.items():
            for tmp_rank in rank_p:
                if tmp_rank not in p_to_d.keys():
                    p_to_d[tmp_rank] = []
                p_to_d[tmp_rank].append(rank_d)
        return p_to_d

    def clear_remote_info(self):
        self.p_inst_enable_sp_cp = False
        self.remote_sp_size = 0
        self.remote_cp_size = 0

        self.remote_link_cluster_id = {}
        self.remote_link_host_ip = {}
        self.remote_link_device_ips = {}
        self.remote_link_device_physical_id = {}

        self.remote_unlink_cluster_id = {}
        self.remote_unlink_device_ips = {}

    def init_dmi_config(self):
        self.role = self.parse("role", required=True)
        if self.role != DmiModeNodeRole.PREFILL and self.role != DmiModeNodeRole.DECODER and\
            self.role != DmiModeNodeRole.FLEX:
            error_msg = "The pd_role should be prefill or decoder in DMI mode."
            logger.error(error_msg)
            raise ValueError(error_msg)

        device_logical_id = self.model_config["local_logic_device_id"].split(",")
        device_physical_id = self.model_config["local_physical_device_id"].split(",")

        model_rank = self.local_rank if self.distributed_enable else self.rank
        if "local_super_device_id" in self.model_config.keys():
            super_device_id = self.model_config["local_super_device_id"].split(",")
            self.model_config["local_super_device_id"] = super_device_id[model_rank]
        local_device_ip = self.model_config["local_device_ip"].split(",")[model_rank]
        self.model_config["npu_device_id"] = device_logical_id[self.local_rank]

        if self.global_world_size != 0:
            self.model_config["world_size"] = self.global_world_size
        else:
            self.model_config["world_size"] = str(len(device_logical_id))
        self.model_config["rank"] = self.rank

        local_host_ip = self.model_config["local_host_ip"].split(",")
        node_num = self.rank * len(local_host_ip) // int(self.model_config["world_size"])
        self.model_config["local_host_ip"] = local_host_ip[node_num]

        if self.dp_size > 1:
            self.local_dp_rank_to_id = generate_dp_inst_id(self.model_config['local_instance_id'], self.dp_size)
            self.model_config["local_instance_id"] = self.local_dp_rank_to_id[self.rank // self.tp_size]
        self.model_config["local_device_ip"] = local_device_ip
        self.model_config["local_physical_device_id"] = device_physical_id[self.local_rank]

        log_msg = (
            f"[Config]>>> local link info:"
            f"global_rank = {self.rank},"
            f"local_host_ip = {self.model_config['local_host_ip']},"
            f"local_instance_id = {self.model_config['local_instance_id']},"
            f"local_device_ip = {self.model_config['local_device_ip']},"
            f"local_logic_device_id = {self.model_config['npu_device_id']},"
            f"local_physical_device_id = {self.model_config['local_physical_device_id']}."
        )
        logger.info(log_msg)

    def set_pd_role(self, role):
        if role == 1:
            self.role = DmiModeNodeRole.PREFILL
        elif role == 2:
            self.role = DmiModeNodeRole.DECODER
        elif role == 3:
            self.role = DmiModeNodeRole.FLEX
        else:
            self.role = "unknown"
        logger.info("[Config]\t>>> Reset PD role finish. PD role is: %s", self.role)

    def set_pd_link_info(self, requests):
        logger.info("[Config]\t>>> start to set PD link/unlink info according to the request.")
        self.clear_remote_info()

        attr_info = np.array(requests[ATTRIBUTES_IDX], dtype=np.int64, copy=False)
        device_info = np.array(requests[DEVICES_IDX], dtype=np.int64, copy=False)
        policy_info = np.array(requests[POLICY_IDX], dtype=np.int64, copy=False)
        logger.info("[Config]>>> PD remote attr info: %s.", attr_info.tolist())
        logger.info("[Config]>>> PD remote link/unlink info: %s.", device_info.tolist())
        logger.info("[Config]>>> PD remote policy info: %s.", policy_info.tolist())

        self.set_pd_role(attr_info[0][0])
        self.need_switch = (attr_info[0][1] == 1)
        link_num = attr_info[0][2]
        unlink_num = attr_info[0][3]
        host_ip_num = attr_info[0][4]
        super_id_num = attr_info[0][5]
        contains_dpinstance_ids = attr_info[0][6]

        # 暂时只支持所有remote实例相同tp dp sp配置
        self.remote_sp_size = int(policy_info[0][1])
        self.remote_cp_size = int(policy_info[0][2])
            
        self.p_inst_enable_sp_cp = ((self.role == DmiModeNodeRole.PREFILL) and (self.sp_size * self.cp_size != 1)) or \
                                ((self.role == DmiModeNodeRole.DECODER) and (self.remote_sp_size * self.remote_cp_size != 1))

        logger.info("[Config]\t>>> PD switch is %s, link num : %s, unlink_num : %s," \
                    " host_ip_num : %s, p_inst_enable_sp_cp : %s.",
                    self.need_switch, link_num, unlink_num, host_ip_num, self.p_inst_enable_sp_cp)

        if link_num == 0 and unlink_num == 0:
            logger.info("[Config]\t>>> Do not need to link and unlink.")
            return

        # dp * tp * cp should be equal to worldsize
        if self.dp_size * self.tp_size * self.cp_size != int(self.model_config["world_size"]):
            error_msg = (
                f"[Config]\t>>> Unsupport model parallel strategy, "
                f"dp: {self.dp_size}, tp: {self.tp_size}, cp: {self.cp_size}, "
                f"world size: {self.model_config['world_size']}, tp * dp * cp != world_size."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # generate prefill<--->decoder link map
        # link_map will update at every link time (compatiable for pd role switch)
        # 判断是否存在跨机的情况
        is_cross_machine = (host_ip_num > link_num) and contains_dpinstance_ids == 1
        host_ip_per_dp = host_ip_num // link_num
        
        logger.info(f"[Config]\t>>> Is remote dp group across machine: {is_cross_machine}.")
        if is_cross_machine:
            # 跨机情况下，需要考虑多个 host IP，tensor 结构发生变化
            device_count_per_dp = device_info.shape[1] - host_ip_per_dp
            remote_tp_size = int(device_count_per_dp // self.remote_cp_size)
        else:
            remote_tp_size = int((device_info.shape[1] - 1) // self.remote_cp_size)

        tp_p = self.tp_size if self.role == DmiModeNodeRole.PREFILL else remote_tp_size
        tp_d = self.tp_size if self.role == DmiModeNodeRole.DECODER else remote_tp_size
        sp_p = self.sp_size if self.role == DmiModeNodeRole.PREFILL else self.remote_sp_size
        sp_d = self.sp_size if self.role == DmiModeNodeRole.DECODER else self.remote_sp_size
        cp_p = self.cp_size if self.role == DmiModeNodeRole.PREFILL else self.remote_cp_size
        cp_d = self.cp_size if self.role == DmiModeNodeRole.DECODER else self.remote_cp_size

        params = LinkMapParams(role=self.role, tp_p=tp_p, tp_d=tp_d, sp_p=sp_p, sp_d=sp_d, cp_p=cp_p, cp_d=cp_d)
        self.link_map = DmiConfig.generate_link_map(params)
        info_msg = (
            f"[Config]\t>>> role: {self.role}, "
            f"tp_p: {tp_p}, tp_d: {tp_d}, sp_p: {sp_p}, sp_d: {sp_d}, cp_p: {cp_p}, cp_d: {cp_d}, "
            f"using link map: {self.link_map}"
        )
        logger.info(info_msg)

        # link info
        rank_mode = self.rank % self.tp_size
        # some devices do not neet to link/unlink, early return
        if rank_mode not in self.link_map.keys():
            logger.info(f"Role {self.role}, global_rank {self.rank} do not link with remote instances.")
            return
        remote_link_ranks = self.link_map[rank_mode]

        for i in range(0, link_num):
            link_device = device_info[i]
            dp_instance_id = int(link_device[0][8])
            device_start = host_ip_per_dp if is_cross_machine else 1
            instance_id = dp_instance_id if self.dp_size == 1 else dp_instance_id // 10000
            if instance_id not in self.remote_link_cluster_id.keys():
                self.remote_link_cluster_id[instance_id] = []
                self.remote_link_device_ips[instance_id] = []
                self.remote_link_device_physical_id[instance_id] = []
                if host_ip_num != 0:
                    self.remote_link_host_ip[instance_id] = []
                if super_id_num != 0:
                    self.remote_super_pod_id[instance_id] = []
                    self.remote_super_device_id[instance_id] = []

            for remote_rank in remote_link_ranks:
                # for one device links with several devices, mark every link using different cluster id
                cp_remote_rank = remote_rank // self.remote_sp_size
                sp_remote_rank = remote_rank % self.remote_sp_size
                if self.p_inst_enable_sp_cp:
                    # save 3 digits for cp and 2 digits for sp
                    cluster_id = dp_instance_id * 100000 + cp_remote_rank * 100 + sp_remote_rank
                else:
                    cluster_id = dp_instance_id

                self.dp_inst_id_to_cluster_id.setdefault(dp_instance_id, []).append(cluster_id)
                # 计算实际的 device 索引
                device_index = device_start + remote_rank
                link_device_ip = ip_array_to_string(link_device[device_index][:8])
                self.remote_link_device_ips[instance_id].append(link_device_ip)
                self.remote_link_cluster_id[instance_id].append(cluster_id)
                self.remote_link_device_physical_id[instance_id].append(int(link_device[device_index][8]))
                if host_ip_num != 0:
                    if is_cross_machine:
                        # 跨机情况下，根据 device 位置选择对应的 host IP，当计数超过 remote_tp_size 时，host_index+1
                        host_index = remote_rank // remote_tp_size
                        self.remote_link_host_ip[instance_id].append(ip_array_to_string(link_device[host_index][:8]))
                    else:
                        self.remote_link_host_ip[instance_id].append(ip_array_to_string(link_device[0][:8]))
                if super_id_num != 0:
                    self.remote_super_pod_id[instance_id].append(int(link_device[0][9]))
                    self.remote_super_device_id[instance_id].append(int(link_device[device_index][9]))
        log_msg = (
            f"[Config]>>> link msg:"
            f"global_rank = {self.rank},"
            f"local_instance_id = {self.model_config['local_instance_id']},"
            f"local_device_ip = {self.model_config['local_device_ip']};"
            f"remote_link_cluster_id = {self.remote_link_cluster_id},"
            f"remote_link_device_physical_id = {self.remote_link_device_physical_id},"
            f"remote_link_host_ip = {self.remote_link_host_ip},"
            f"remote_super_pod_id = {self.remote_super_pod_id},"
            f"remote_super_device_id = {self.remote_super_device_id},"
            f"remote_link_device_ips = {self.remote_link_device_ips}."
        )
        logger.info(log_msg)

        # unlink info
        for i in range(link_num, link_num + unlink_num):
            unlink_device = device_info[i]
            dp_instance_id = int(unlink_device[0][8])
            device_start = host_ip_per_dp if is_cross_machine else 1
            instance_id = dp_instance_id if self.dp_size == 1 else dp_instance_id // 10000
            if instance_id not in self.remote_unlink_cluster_id.keys():
                self.remote_unlink_device_ips[instance_id] = []
                self.remote_unlink_cluster_id[instance_id] = []

            for remote_rank in remote_link_ranks:
                cp_remote_rank = remote_rank // self.remote_sp_size
                sp_remote_rank = remote_rank % self.remote_sp_size
                if self.p_inst_enable_sp_cp:
                    cluster_id = dp_instance_id * 100000 + cp_remote_rank * 100 + sp_remote_rank
                else:
                    cluster_id = dp_instance_id
                device_index = device_start + remote_rank
                unlink_device_ip = ip_array_to_string(unlink_device[device_index][:8])
                self.remote_unlink_device_ips[instance_id].append(unlink_device_ip)
                self.remote_unlink_cluster_id[instance_id].append(cluster_id)
        log_msg = (
            f"[Config]>>> unlink msg:"
            f"global_rank = {self.rank},"
            f"local_instance_id = {self.model_config['local_instance_id']},"
            f"local_device_ip = {self.model_config['local_device_ip']};"
            f"remote_unlink_cluster_id = {self.remote_unlink_cluster_id},"
            f"remote_unlink_device_ips = {self.remote_unlink_device_ips}."
        )
        logger.info(log_msg)