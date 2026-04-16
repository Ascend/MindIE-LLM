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
import time
import threading
from collections import deque

from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
from llm_datadist import LLMRole, BlocksCacheKey, LLMStatusCode
from llm_datadist import CacheDesc, DataType, LLMException, Placement
from llm_datadist import LLMDataDist, RegisterMemStatus
from llm_datadist import LLMConfig as LLMDataDistConfig
from mindie_llm.utils.log.logging import logger
from mindie_llm.utils.log.error_code import ErrorCode
from mindie_llm.utils.status import MindieLlmStatusCode
from mindie_llm.model_wrapper.utils.common_util import get_ip_obj


STR_DTYPE_TO_DTYPE = {
    "float16": DataType.DT_FLOAT16,
    "bfloat16": DataType.DT_BF16,
    "float": DataType.DT_FLOAT,
    "int8": DataType.DT_INT8,
}


class LinkResult:
    def __init__(self):
        self._waiting_links = []
        self._running_links = []
        self._failed_links = []
        self._success_links = []
        self._lock = threading.Lock()

    def add_to_waiting(self, link_ip):
        with self._lock:
            self.remove_from_current_list(link_ip)
            self._waiting_links.append(link_ip)

    def add_to_running(self, link_ip):
        with self._lock:
            self.remove_from_current_list(link_ip)
            self._running_links.append(link_ip)

    def add_to_failed(self, link_ip):
        with self._lock:
            self.remove_from_current_list(link_ip)
            self._failed_links.append(link_ip)

    def add_to_success(self, link_ip):
        with self._lock:
            self.remove_from_current_list(link_ip)
            self._success_links.append(link_ip)

    def get_all_status(self) -> Dict[str, List[str]]:
        """获取所有链接的状态汇总"""
        with self._lock:
            return {
                "waiting": self._waiting_links.copy(),
                "running": self._running_links.copy(),
                "success": self._success_links.copy(),
                "failed": self._failed_links.copy(),
            }

    def remove_from_current_list(self, link_ip):
        current_list = self._get_current_list(link_ip)
        if current_list is not None and link_ip in current_list:
            current_list.remove(link_ip)

    def _get_current_list(self, link_ip) -> Optional[list]:
        """
        查找元素当前所在的链表，返回该链表，无则返回None。
        """
        if link_ip in self._waiting_links:
            return self._waiting_links
        elif link_ip in self._running_links:
            return self._running_links
        elif link_ip in self._failed_links:
            return self._failed_links
        elif link_ip in self._success_links:
            return self._success_links
        return None


class DmiModeNodeRole(str, Enum):
    """
    PD分离特性中的节点角色管理机制。
    在PD分离架构中,不同类型的计算节点被赋予特定角色以优化处理效率:
    - Flex(弹性)节点:具备动态任务处理能力,可根据系统负载同时处理Prefill和Decode请求
    - 角色细分：
    * FlexP   - 专用于PD分离Prefill阶段请求处理
    * FlexD   - 专用于PD分离Decode阶段请求处理
    * FlexPnD - 支持Prefill和Decode在同一个节点上请求处理
    """

    """节点工作模式角色枚举"""
    PREFILL = "prefill"  # 专用于请求预处理阶段（Prefill）的节点角色
    DECODER = "decoder"  # 专用于序列解码阶段（Decode）的节点角色
    FLEX = "flex"  # 弹性节点角色，可动态处理Prefill和Decode混合请求


@dataclass
class LinkParams:
    remote_cluster_ids: Dict[int, List[int]]
    remote_physical_device_ids: Dict[int, List[int]]
    remote_device_ips: Dict[int, List[str]]
    host_ips: Dict[int, List[str]]
    remote_super_device_ids: Dict[int, List[int]] | None = None
    remote_super_pod_ids: Dict[int, List[int]] | None = None


class RankInfo:
    def __init__(
        self,
        remote_cluster_id,
        remote_physical_device_id,
        remote_device_ip,
        remote_host_ip,
        local_host_ip,
        local_cluster_id,
        local_device_ip,
        local_physical_device_id,
        local_super_device_id=None,
        remote_super_device_id=None,
        local_super_pod_id=None,
        remote_super_pod_id=None,
    ):  # 保留链接索引参数
        self.local_cluster_id = local_cluster_id
        self.remote_cluster_id = remote_cluster_id
        self.remote_physical_device_id = str(remote_physical_device_id)
        self.remote_device_ip = remote_device_ip
        self.remote_server_id = remote_host_ip
        self.local_server_id = local_host_ip
        self.local_device_ip = local_device_ip
        self.local_physical_device_id = str(local_physical_device_id)
        rank_ids = self.assign_rank_id(remote_device_ip, local_device_ip)
        self.local_rank_id = rank_ids[local_device_ip]
        self.remote_rank_id = rank_ids[remote_device_ip]
        self.local_super_device_id = local_super_device_id
        self.remote_super_device_id = remote_super_device_id
        self.local_super_pod_id = local_super_pod_id
        self.remote_super_pod_id = remote_super_pod_id

    @staticmethod
    def assign_rank_id(remote_device_ip: str, local_device_ip: str) -> Dict[str, int]:
        # 比较 ip 大小分配 rank 地址，小卡用"0"，大卡用"1"
        remote_device_ip_obj = get_ip_obj(remote_device_ip, "remote_device_ip")
        local_device_ip_obj = get_ip_obj(local_device_ip, "local_device_ip")

        if remote_device_ip_obj < local_device_ip_obj:
            return {remote_device_ip: 0, local_device_ip: 1}
        else:
            return {remote_device_ip: 1, local_device_ip: 0}

    @staticmethod
    def create_device(device_id, device_ip, rank_id, super_device_id=None) -> Dict:
        device_info = {
            "device_id": device_id,
            "device_ip": device_ip,
            "rank_id": str(rank_id),
        }
        if super_device_id is not None:
            device_info["super_device_id"] = str(super_device_id)
        return device_info

    def create_super_pod_list(self):
        super_pod_list = []
        server_id_key = "server_id"
        server_list_key = "server_list"
        super_pod_id_key = "super_pod_id"
        server_list = [{server_id_key: str(self.local_server_id)}]
        if self.local_super_pod_id == self.remote_super_pod_id:
            if self.local_server_id != self.remote_server_id:
                server_list.append({server_id_key: str(self.remote_server_id)})

            super_pod_list = [
                {
                    super_pod_id_key: str(self.local_super_pod_id),
                    server_list_key: server_list,
                }
            ]
        else:
            super_pod_list = [
                {
                    super_pod_id_key: str(self.local_super_pod_id),
                    server_list_key: [{server_id_key: str(self.local_server_id)}],
                },
                {
                    super_pod_id_key: str(self.remote_super_pod_id),
                    server_list_key: [{server_id_key: str(self.remote_server_id)}],
                },
            ]
        return super_pod_list

    def get_rank_table(self) -> str:
        rank_table_status = "completed"
        rank_table_version = "1.0"
        device = "device"
        server_id = "server_id"
        server_list: List[Dict] = []

        if self.remote_server_id == self.local_server_id:
            # 当远程服务器ID等于本地服务器ID时，只需要一个服务器条目，但包含两个设备
            server = {
                device: [
                    self.create_device(
                        self.local_physical_device_id,
                        self.local_device_ip,
                        self.local_rank_id,
                        self.local_super_device_id,
                    ),
                    self.create_device(
                        self.remote_physical_device_id,
                        self.remote_device_ip,
                        self.remote_rank_id,
                        self.remote_super_device_id,
                    ),
                ],
                server_id: str(self.local_server_id),
            }
            server_list.append(server)
            server_count = "1"
        else:
            # 当远程服务器ID不等于本地服务器ID时，有两个服务器条目，每个服务器包含一个设备
            # A3限制必须对端和本端的rank table的server list一样
            if self.local_rank_id < self.remote_rank_id:
                server_list.append(
                    {
                        device: [
                            self.create_device(
                                self.local_physical_device_id,
                                self.local_device_ip,
                                self.local_rank_id,
                                self.local_super_device_id,
                            )
                        ],
                        server_id: str(self.local_server_id),
                    }
                )
                server_list.append(
                    {
                        device: [
                            self.create_device(
                                self.remote_physical_device_id,
                                self.remote_device_ip,
                                self.remote_rank_id,
                                self.remote_super_device_id,
                            )
                        ],
                        server_id: str(self.remote_server_id),
                    }
                )
            else:
                server_list.append(
                    {
                        device: [
                            self.create_device(
                                self.remote_physical_device_id,
                                self.remote_device_ip,
                                self.remote_rank_id,
                                self.remote_super_device_id,
                            )
                        ],
                        server_id: str(self.remote_server_id),
                    }
                )
                server_list.append(
                    {
                        device: [
                            self.create_device(
                                self.local_physical_device_id,
                                self.local_device_ip,
                                self.local_rank_id,
                                self.local_super_device_id,
                            )
                        ],
                        server_id: str(self.local_server_id),
                    }
                )
            server_count = "2"
        rank_table_dict = {
            "server_count": server_count,
            "server_list": server_list,
            "status": rank_table_status,
            "version": rank_table_version,
        }
        if self.local_super_device_id is not None:
            rank_table_version = "1.2"
            rank_table_dict["version"] = rank_table_version
            super_pod_list = self.create_super_pod_list()
            rank_table_dict["super_pod_list"] = super_pod_list

        return json.dumps(rank_table_dict)

    def get_cluster_rank_info(self) -> dict:
        if self.local_rank_id < self.remote_rank_id:
            return {
                int(self.local_cluster_id): self.local_rank_id,
                int(self.remote_cluster_id): self.remote_rank_id,
            }
        else:
            return {
                int(self.remote_cluster_id): self.remote_rank_id,
                int(self.local_cluster_id): self.local_rank_id,
            }


class SeparateDeploymentEngine:
    def __init__(
        self,
        role=DmiModeNodeRole.DECODER,
        local_cluster_id=0,
        local_logic_device_id=0,
        kv_trans_timeout=1,
        kv_rdma_sl=-1,
        kv_rdma_tc=-1,
        kv_link_timeout=1080,
    ):
        engine_role = None
        if role == DmiModeNodeRole.PREFILL:
            engine_role = LLMRole.PROMPT
        elif role == DmiModeNodeRole.DECODER:
            engine_role = LLMRole.DECODER
        elif (
            role == DmiModeNodeRole.FLEX
        ):  # 底层LLMDataDist接口给出的枚举是MIX，上层定义的微调实例名称为flex，在此处进行转换
            engine_role = LLMRole.MIX

        self.role = role

        self.separate_deployment_engine = LLMDataDist(engine_role, local_cluster_id)

        llm_config = LLMDataDistConfig()
        llm_config.device_id = local_logic_device_id
        llm_config.enable_cache_manager = True
        llm_config.enable_remote_cache_accessible = True
        llm_config.link_total_time = int(kv_link_timeout)
        llm_config.link_retry_count = 80
        # 配置pull kv 超时时间，默认为1秒，sync_kv_timeout单位为ms。
        if kv_trans_timeout <= 0:
            kv_trans_timeout = 1
        llm_config.sync_kv_timeout = kv_trans_timeout * 1000
        logger.info(
            f"kv_trans_timeout: {llm_config.sync_kv_timeout}."
            f"link_total_time: {llm_config.link_total_time}."
            f"link_retry_count: {llm_config.link_retry_count}."
        )
        llm_options = llm_config.generate_options()
        if kv_rdma_sl != -1 and kv_rdma_tc != -1:
            if kv_rdma_sl < 0 or kv_rdma_sl > 7:
                raise Exception("SeparateDeploymentEngine: kv_rdma_sl only support: 0-7.")
            if kv_rdma_tc < 0 or kv_rdma_tc > 255:
                raise Exception("SeparateDeploymentEngine: kv_rdma_tc only support: 0-255.")
            llm_options["llm.RdmaServiceLevel"] = str(kv_rdma_sl)
            llm_options["llm.RdmaTrafficClass"] = str(kv_rdma_tc)
            logger.info(
                f"kv_rdma_sl: {llm_options['llm.RdmaServiceLevel']}, kv_rdma_tc: {llm_options['llm.RdmaTrafficClass']}."
            )
        self.separate_deployment_engine.init(llm_options)
        self.npu_tensors = []
        self.npu_cache_map = {}

    def link(self, cluster_rank_info: Dict[int, int], rank_table: str):
        rank_table_dict = json.loads(rank_table)
        server_count = rank_table_dict.get("server_count")
        server_list = rank_table_dict.get("server_list", [])
        device = "device"
        device_ip = "device_ip"
        status = "status"
        comm_id = "comm_id"
        device_ip1 = None
        device_ip2 = None
        if server_count == "1":
            server = server_list[0]
            devices = server.get(device, [])
            device_ip1 = devices[0].get(device_ip)
            device_ip2 = devices[1].get(device_ip)
        elif server_count == "2":
            server1 = server_list[0]
            server2 = server_list[1]
            devices1 = server1.get(device, [])
            devices2 = server2.get(device, [])
            device_ip1 = devices1[0].get(device_ip)
            device_ip2 = devices2[0].get(device_ip)

        if device_ip1 > device_ip2:
            link_name = f"link{device_ip1}:{device_ip2}"
        else:
            link_name = f"link{device_ip2}:{device_ip1}"

        logger.info(
            f"Link params cluster_rank_info: {cluster_rank_info}, rank_table: {rank_table}, link_name: {link_name}."
        )

        try:
            result = self.separate_deployment_engine.link(link_name, cluster_rank_info, rank_table)
            # 检查返回值是否为已建链错误码
            if result == LLMStatusCode.LLM_ALREADY_LINK:
                return {
                    status: MindieLlmStatusCode.TEXT_GENERATOR_PD_ALREADY_LINK,
                    comm_id: None,
                }
            # 正常返回通信ID
            return {status: MindieLlmStatusCode.SUCCESS, comm_id: result}
        except LLMException as e:
            logger.error(
                f"Link failed, error code is {e.status_code}, rank_table is {rank_table}, "
                f"cluster_rank_info is {cluster_rank_info}."
            )
            return {status: ErrorCode.TEXT_GENERATOR_PD_LINK_ERROR, comm_id: None}

    def unlink(self, comm_id: int):
        try:
            self.separate_deployment_engine.unlink(comm_id)
            return MindieLlmStatusCode.SUCCESS
        except LLMException:
            logger.error(f"Unlink failed, error code is {ErrorCode.TEXT_GENERATOR_PD_UNLINK_ERROR}.")
            return ErrorCode.TEXT_GENERATOR_PD_UNLINK_ERROR

    def set_npu_cache(self, model_id, npu_cache):
        self.npu_cache_map[model_id] = npu_cache

    def pull_kv(
        self,
        model_id: int,
        src_block_table: List[int],
        dst_block_table: List[int],
        remote_cluster_id: int,
    ):
        remote_cache_key = BlocksCacheKey(cluster_id=remote_cluster_id, model_id=model_id)
        try:
            self.separate_deployment_engine.cache_manager.pull_blocks(
                remote_cache_key,
                self.npu_cache_map[model_id],
                src_block_table,
                dst_block_table,
            )
            return MindieLlmStatusCode.SUCCESS
        except LLMException as e:
            logger.error(
                f"Pull kv from remote_cluster_id: {remote_cluster_id} failed, CANN status_code is: {e.status_code}."
            )
            return ErrorCode.TEXT_GENERATOR_PD_PULL_KV_ERROR

    def register_blocks_cache(self, cache_desc, npu_addrs, cache_key):
        cache_manager = self.separate_deployment_engine.cache_manager
        return cache_manager.register_blocks_cache(cache_desc, npu_addrs, cache_key)

    def query_register_mem_status(self, comm_id: int):
        return self.separate_deployment_engine.query_register_mem_status(comm_id)

    def finalize(self):
        self.separate_deployment_engine.finalize()


class SeparateDeploymentWorker:
    def __init__(
        self,
        role: str,
        local_logic_device_id: int,
        local_physical_device_id: int,
        local_cluster_id: int,
        local_device_ip: str,
        local_host_ip: str,
        local_super_pod_id: int | None = None,
        local_super_device_id: int | None = None,
        kv_trans_timeout: int = 1,
        kv_link_timeout: int = 1080,
        kv_rdma_sl: int = -1,
        kv_rdma_tc: int = -1,
    ):
        self.role = role
        self.local_logic_device_id = local_logic_device_id
        self.local_physical_device_id = local_physical_device_id
        self.local_cluster_id = local_cluster_id
        self.local_device_ip = local_device_ip
        self.local_host_ip = local_host_ip
        self.local_super_device_id = local_super_device_id
        self.local_super_pod_id = local_super_pod_id
        self.cluster_comm_map = {}
        self.cluster_device_ip_map = {}  # 新增：cluster_id到device_ip的映射
        self.link_time_out = kv_link_timeout
        if self.link_time_out <= 0:
            self.link_time_out = 1080
        logger.info(f"kv_link_timeout: {self.link_time_out}.")
        self.try_create_link_time_out = 5 * 60
        self.link_start_time = 0

        self.window_size = 16
        self.window = deque()  # 将window声明为成员变量
        self.link_queue = deque()

        self.link_queue_cond = threading.Condition()  # 保护link_queue
        self.window_cond = threading.Condition()  # 保护window
        self.cluster_map_lock = threading.Lock()  # 保护cluster相关映射

        self.link_result = LinkResult()
        self.cache_desc_map = {}
        self.max_block_nums_map = {}
        if role in [
            DmiModeNodeRole.DECODER,
            DmiModeNodeRole.PREFILL,
            DmiModeNodeRole.FLEX,
        ]:
            self.separate_deployment_engine = SeparateDeploymentEngine(
                role=role,
                local_cluster_id=local_cluster_id,
                local_logic_device_id=local_logic_device_id,
                kv_trans_timeout=kv_trans_timeout,
                kv_rdma_sl=kv_rdma_sl,
                kv_rdma_tc=kv_rdma_tc,
                kv_link_timeout=kv_link_timeout,
            )
        else:
            raise Exception("SeparateDeploymentEngine: not support role.")

        # 启动两个工作线程
        self._fill_thread = threading.Thread(target=self._fill_window_worker, name="fill_window_thread", daemon=True)
        self._process_thread = threading.Thread(
            target=self._process_window_worker,
            name="process_window_thread",
            daemon=True,
        )
        self._fill_thread.start()
        self._process_thread.start()

    @staticmethod
    def _create_link_params(instance_id, index, params):
        single_link_params = {
            "remote_cluster_id": params.remote_cluster_ids[instance_id][index],
            "remote_physical_device_id": params.remote_physical_device_ids[instance_id][index],
            "remote_device_ip": params.remote_device_ips[instance_id][index],
            "remote_host_ip": params.host_ips[instance_id][index],
            "retry_count": 0,
        }
        if params.remote_super_device_ids is not None:
            single_link_params["remote_super_device_id"] = params.remote_super_device_ids[instance_id][index]
        if params.remote_super_pod_ids is not None:
            single_link_params["remote_super_pod_id"] = params.remote_super_pod_ids[instance_id][index]

        return single_link_params

    def build(self, model_id: int, num_tensors, num_blocks, blockshape, dtype):
        block_shape = tuple(map(int, blockshape))
        self.cache_desc_map[model_id] = CacheDesc(
            num_tensors=int(num_tensors),
            shape=(int(num_blocks), *block_shape),
            data_type=STR_DTYPE_TO_DTYPE[dtype],
            placement=Placement.DEVICE,
        )
        self.max_block_nums_map[model_id] = num_blocks

    # K npu_cache model_id is 0, V npu_cache model_id is 1
    def set_npu_cache(self, model_id: int, npu_addrs: List[int]):
        try:
            cache_key = BlocksCacheKey(cluster_id=self.local_cluster_id, model_id=model_id)
            npu_cache = self.separate_deployment_engine.register_blocks_cache(
                self.cache_desc_map[model_id], npu_addrs, cache_key
            )
            self.separate_deployment_engine.set_npu_cache(model_id, npu_cache)
            logger.info("Register blocks cache success.")
        except Exception as e:
            logger.error(f"Failed to register blocks cache:{e}.")
            raise Exception("Failed to register blocks cache") from e

    def pull_blocks(
        self,
        remote_model_instance_id: int,
        src_block_table: List[int],
        dst_block_table: List[int],
    ):
        with self.cluster_map_lock:
            if remote_model_instance_id not in self.cluster_comm_map:
                logger.error(f"Pull_kv error: remote_model_instance_id: {remote_model_instance_id} is not linked.")
                return ErrorCode.TEXT_GENERATOR_PD_MODEL_INSTANCE_ID_ERROR

        # 遍历所有注册的model_id,检查block的范围并拉取对应kv cache
        for model_id in self.cache_desc_map:
            if not all(0 <= x < self.max_block_nums_map[model_id] for x in dst_block_table):
                logger.error(f"Pull_kv error: block id out of range, model_id is {model_id}.")
                return ErrorCode.TEXT_GENERATOR_PD_BLOCK_ID_OUT_OF_RANGE
            rt = self.separate_deployment_engine.pull_kv(
                model_id=model_id,
                src_block_table=src_block_table,
                dst_block_table=dst_block_table,
                remote_cluster_id=remote_model_instance_id,
            )
            if rt != MindieLlmStatusCode.SUCCESS:
                # 获取对端device_ip进行详细日志记录
                with self.cluster_map_lock:
                    remote_device_ip = self.cluster_device_ip_map.get(remote_model_instance_id, "unknown")
                logger.error(f"{self.local_device_ip} pull kv from {remote_device_ip} failed, error code is {rt}.")
                return rt
        return MindieLlmStatusCode.SUCCESS

    def query_link_status(self):
        return self.link_result.get_all_status()

    def link(self, **kwargs):
        """
        外部调用接口，将链接任务添加到队列，不阻塞
        """
        # 使用 LinkParams 处理参数
        params = LinkParams(**kwargs)
        instance_ids = list(params.remote_cluster_ids.keys())

        # 将链接任务添加到临时队列中
        link_items = []
        for instance_id in instance_ids:
            for index in range(len(params.remote_cluster_ids[instance_id])):
                link_item = self._create_link_params(instance_id, index, params)
                link_items.append(link_item)

        # 加锁仅保护队列写入，减少锁持有时间
        with self.link_queue_cond:
            for link_item in link_items:
                self.link_queue.append(link_item)
                self.link_result.add_to_waiting(link_item["remote_device_ip"])
            self.link_queue_cond.notify()

    def unlink_batch(self, remote_cluster_ids: List[int]) -> Dict[int, Union[MindieLlmStatusCode, ErrorCode]]:
        """
        批量处理unlink请求
        """
        if not remote_cluster_ids:
            logger.warning("unlink_batch called with empty remote_cluster_ids list")
            return {}

        target_cluster_ids = set(remote_cluster_ids)
        batch_result_map = {}
        removed_from_queue = set()

        with self.link_queue_cond:
            to_remove = []
            for link_item in self.link_queue:
                cluster_id = link_item["remote_cluster_id"]
                if cluster_id in target_cluster_ids:
                    to_remove.append(link_item)
                    removed_from_queue.add(cluster_id)

            for link_item in to_remove:
                self.link_queue.remove(link_item)
                self.link_result.remove_from_current_list(link_item["remote_device_ip"])
                logger.info(
                    f"Batch removed {link_item['remote_device_ip']} "
                    f"(cluster_id: {link_item['remote_cluster_id']}) from waiting list"
                )

            for cluster_id in removed_from_queue:
                batch_result_map[cluster_id] = MindieLlmStatusCode.SUCCESS

            if removed_from_queue:
                self.link_queue_cond.notify()

        remaining_cluster_ids = target_cluster_ids - removed_from_queue
        for cluster_id in remaining_cluster_ids:
            result = self._unlink_after_queue_cleanup(cluster_id)
            batch_result_map[cluster_id] = result

        return batch_result_map

    def unlink(self, remote_cluster_id):
        # 检查是否在等待队列中
        with self.link_queue_cond:
            target_item = None
            target_idx = -1

            for idx, link_item in enumerate(self.link_queue):
                if link_item["remote_cluster_id"] == remote_cluster_id:
                    target_item = link_item
                    target_idx = idx
                    break

            if target_item is not None:
                self.link_result.remove_from_current_list(target_item["remote_device_ip"])
                del self.link_queue[target_idx]
                logger.info(f"Removed {target_item['remote_device_ip']} from waiting list")
                self.link_queue_cond.notify()
                return MindieLlmStatusCode.SUCCESS

        return self._unlink_after_queue_cleanup(remote_cluster_id)

    def unlink_all(self):
        with self.link_queue_cond:
            self.link_queue.clear()

        with self.window_cond:
            self.window.clear()

        for cluster_id in list(self.cluster_comm_map):
            ret = self.unlink(cluster_id)
            if ret != MindieLlmStatusCode.SUCCESS:
                logger.error(f"Failed to unlink cluster {cluster_id}, error code is {ret}.")
                raise Exception("SeparateDeploymentEngine: unlink_all fail")

    def finalize(self):
        self.unlink_all()
        self.separate_deployment_engine.finalize()

    def _try_create_link(self, link_params):
        """
        尝试创建一个链接，支持重试机制

        参数:
            link_params: 链接参数

        返回:
            (connect_success, window_item): connect_success: 标志是否成功，window_item: 失败时为None,成功时返回一个窗口项
        """
        try:
            # 创建RankInfo对象
            rank_info = RankInfo(
                remote_cluster_id=link_params["remote_cluster_id"],
                remote_physical_device_id=link_params["remote_physical_device_id"],
                remote_device_ip=link_params["remote_device_ip"],
                remote_host_ip=link_params["remote_host_ip"],
                local_host_ip=self.local_host_ip,
                local_cluster_id=self.local_cluster_id,
                local_device_ip=self.local_device_ip,
                local_physical_device_id=self.local_physical_device_id,
                local_super_pod_id=self.local_super_pod_id,
                remote_super_pod_id=link_params.get("remote_super_pod_id"),
                local_super_device_id=self.local_super_device_id,
                remote_super_device_id=link_params.get("remote_super_device_id"),
            )

            cluster_rank_info = rank_info.get_cluster_rank_info()
            rank_table = rank_info.get_rank_table()

            # 尝试链接
            result = self.separate_deployment_engine.link(cluster_rank_info, rank_table)

            # 从结果中提取状态和comm_id
            status = result.get("status")
            comm_id = result.get("comm_id")

            if status == MindieLlmStatusCode.SUCCESS and comm_id is not None:
                with self.cluster_map_lock:
                    self.cluster_comm_map[link_params["remote_cluster_id"]] = comm_id
                    # 创建窗口项
                    window_item = {"comm_id": comm_id, "link_params": link_params}
                    return True, window_item
            else:
                # 通信域初始化失败，status = ErrorCode.TEXT_GENERATOR_PD_LINK_ERROR
                logger.error(
                    f"Link from {self.local_device_ip} to {link_params['remote_device_ip']} failed, "
                    f"error code is {status}."
                )
                return False, None

        except Exception as e:
            logger.error(f"Link exception from {self.local_device_ip} to {link_params['remote_device_ip']}: {str(e)}.")
            return False, None

    def _query_mem_status(self, link_item):
        """
        查询链路状态并处理结果

        参数:
            link_item: 链路项，包含 comm_id, link_params 等信息

        返回:
            status: 状态码
        """
        comm_id = link_item["comm_id"]
        link_params = link_item["link_params"]
        remote_cluster_id = link_params["remote_cluster_id"]

        # 限制查询次数，避免无限轮询
        max_query_attempts = 3
        query_attempt = 0

        while query_attempt < max_query_attempts:
            try:
                # 记录查询开始时间
                query_start_time = time.time()
                logger.debug(f"Querying mem status for {link_params['remote_device_ip']}, attempt {query_attempt + 1}")

                # 尝试查询内存状态
                query_status = self.separate_deployment_engine.query_register_mem_status(comm_id)

                # 记录查询结束时间
                query_end_time = time.time()
                query_duration = query_end_time - query_start_time
                logger.debug(f"Query completed in {query_duration:.2f}s for {link_params['remote_device_ip']}")

                if query_status == RegisterMemStatus.OK:
                    logger.info(
                        f"Link from local_device_ip {self.local_device_ip} "
                        f"to remote_device_ip {link_params['remote_device_ip']} success."
                    )
                    # 建立cluster_id到device_ip的映射
                    with self.cluster_map_lock:
                        self.cluster_device_ip_map[link_params["remote_cluster_id"]] = link_params["remote_device_ip"]
                    return MindieLlmStatusCode.SUCCESS  # 成功

                elif query_status == RegisterMemStatus.FAILED:
                    logger.error(f"Mem status query failed for {link_params['remote_device_ip']}, no retry.")
                    ret = self.unlink(remote_cluster_id)
                    if ret != MindieLlmStatusCode.SUCCESS:
                        logger.error(f"Failed to unlink remote_cluster_id {remote_cluster_id}, error code is {ret}.")
                    return ErrorCode.TEXT_GENERATOR_PD_LINK_ERROR
                else:
                    # 状态为PENDING或其他，继续下一次查询
                    query_attempt += 1
                    # 短暂等待后再次查询
                    if query_attempt < max_query_attempts:
                        time.sleep(0.05)

            except Exception as e:
                logger.error(f"Mem status query failed for {link_params['remote_device_ip']} with exception{e}.")
                ret = self.unlink(remote_cluster_id)
                if ret != MindieLlmStatusCode.SUCCESS:
                    logger.error(f"Failed to unlink remote_cluster_id {remote_cluster_id}, error code is {ret}.")
                return ErrorCode.TEXT_GENERATOR_PD_LINK_ERROR

        # 达到最大查询次数，但状态仍然不确定，重新加入队列稍后处理
        logger.debug(f"Max query attempts reached for {link_params['remote_device_ip']}, requeueing")
        return MindieLlmStatusCode.TEXT_GENERATOR_PD_RETRY_QUERY

    def _fill_window_worker(self):
        """
        填充窗口线程:持续从link_queue获取任务尝试，创建链接并填充到窗口
        """
        time_wait = False
        while True:
            # 检查窗口是否已满
            with self.window_cond:
                if len(self.window) >= self.window_size:
                    time_wait = True

            if time_wait:  # 窗口满时短暂休眠
                time_wait = False
                time.sleep(0.1)
                continue
            # 从队列获取任务
            link_item = None
            with self.link_queue_cond:
                # 等待：队列非空 且 窗口未满 时才唤醒
                if not self.link_queue_cond.wait_for(
                    lambda: len(self.link_queue) > 0 and len(self.window) < self.window_size,
                    timeout=0.1,
                ):
                    continue
                link_item = self.link_queue.popleft()

            if not link_item:
                continue

            # 尝试创建链接
            link_status, window_item = self._try_create_link(link_item)

            with self.window_cond:
                if link_status:
                    # 创建成功，加入窗口
                    self.window.append(window_item)
                    self.link_result.add_to_running(link_item["remote_device_ip"])
                else:
                    logger.error(
                        f"Link from {self.local_device_ip} to {link_item['remote_device_ip']} "
                        f"error code is {ErrorCode.TEXT_GENERATOR_PD_LINK_ERROR}."
                    )
                    self.link_result.add_to_failed(link_item["remote_device_ip"])

    def _process_window_worker(self):
        """
        处理窗口线程：持续处理窗口中的任务，检查执行状态
        """
        while True:
            # 从窗口获取任务
            window_item = None
            with self.window_cond:
                if not self.window:
                    self.window_cond.wait(0.1)  # 窗口为空时等待
                    continue
                window_item = self.window.popleft()

            if not window_item:
                continue

            # 查询链接状态
            status = self._query_mem_status(window_item)

            with self.window_cond:
                if status == MindieLlmStatusCode.SUCCESS:
                    self.link_result.add_to_success(window_item["link_params"]["remote_device_ip"])

                elif status == MindieLlmStatusCode.TEXT_GENERATOR_PD_RETRY_QUERY:
                    self.window.append(window_item)

                else:
                    # 执行失败，标记为失败
                    logger.error(
                        f"Link from {self.local_device_ip} to {window_item['link_params']['remote_device_ip']} "
                        f"failed, error code is {ErrorCode.TEXT_GENERATOR_PD_LINK_ERROR}."
                    )
                    self.link_result.add_to_failed(window_item["link_params"]["remote_device_ip"])

    def _unlink_after_queue_cleanup(self, remote_cluster_id):
        # 检查是否在window中（running状态）
        with self.window_cond:
            target_item = None
            target_idx = -1

            for idx, window_item in enumerate(self.window):
                if window_item["link_params"]["remote_cluster_id"] == remote_cluster_id:
                    target_item = window_item
                    target_idx = idx
                    break

            if target_item is not None:
                self.link_result.remove_from_current_list(window_item["link_params"]["remote_device_ip"])
                del self.window[target_idx]
                logger.info(f"Remove {target_item['link_params']['remote_device_ip']} from running list")
                self.window_cond.notify()

        # 不在等待队列和窗口中，尝试直接unlink
        with self.cluster_map_lock:
            if remote_cluster_id not in self.cluster_comm_map:
                logger.error(f"Unlink failed, remote_cluster_id:{remote_cluster_id} is not linked.")
                return ErrorCode.TEXT_GENERATOR_PD_UNLINK_ERROR
            comm_id = self.cluster_comm_map.pop(remote_cluster_id)
            # 同时从device_ip映射中移除
            if remote_cluster_id in self.cluster_device_ip_map:
                self.link_result.remove_from_current_list(self.cluster_device_ip_map[remote_cluster_id])
                self.cluster_device_ip_map.pop(remote_cluster_id, None)
            else:
                logger.warning(
                    f"Cannot find device_ip for remote_cluster_id {remote_cluster_id} in cluster_device_ip_map"
                )

        return self.separate_deployment_engine.unlink(comm_id)
