# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import argparse
import signal
import sys

from mindie_llm.connector.common.gc_monitor import GCMonitor
from mindie_llm.connector.request_listener.request_listener import RequestListener
from mindie_llm.utils.log.logging import logger
from mindie_llm.connector.common.global_variables import ProcessStartArgName
from mindie_llm.connector.common.adaptive_garbage_collector import AdaptiveGarbageCollector


def parse_from_cmd():
    parser = argparse.ArgumentParser()
    hyphens = "--"
    parser.add_argument(hyphens + ProcessStartArgName.LOCAL_RANK, type=int, default=0, help="local_rank")
    parser.add_argument(hyphens + ProcessStartArgName.LOCAL_WORLD_SIZE, type=int, default=1,
                        help="local_world_size")
    parser.add_argument(hyphens + ProcessStartArgName.GLOBAL_RANK, type=int, help="global_rank")
    parser.add_argument(hyphens + ProcessStartArgName.GLOBAL_WORLD_SIZE, type=int,
                        help="global_world_size")
    parser.add_argument(hyphens + ProcessStartArgName.NPU_NUM_PER_DP, type=int, default=1, help="tp * cp")
    parser.add_argument(hyphens + ProcessStartArgName.NPU_DEVICE_ID, type=int, default=0, help="npu_device_id")
    parser.add_argument(hyphens + ProcessStartArgName.PARENT_PID, type=int, default=1, help="parent_pid")
    parser.add_argument(hyphens + ProcessStartArgName.SHM_NAME_PREFIX, type=str, default="/integrated_testing",
                        help="shm_name_prefix")
    parser.add_argument(hyphens + ProcessStartArgName.COMMUNICATION_TYPE, type=str, default="shared_meme",
                        help="communication_type:shared_meme|http")
    parser.add_argument(hyphens + ProcessStartArgName.USE_MOCK_MODEL, type=bool, default=False, help="use_mock_model")

    parser.add_argument(hyphens + ProcessStartArgName.LAYERWISE_DISAGGREGATED, 
                        type=str, 
                        default="false", 
                        help="layerwise_disaggregated")
    parser.add_argument(hyphens + ProcessStartArgName.LAYERWISE_DISAGGREGATED_ROLE_TYPE,
                        type=str, 
                        default="", 
                        help="layerwise_disaggregated_role_type")
    # 解析参数
    args = parser.parse_args()

    return args


def check_config(args):
    if args.local_rank < 0:
        logger.error(f"local_rank error! local_rank={args.local_rank}")
        return False
    if args.local_world_size <= 0:
        logger.error(f"local_world_size error! local_world_size={args.local_world_size}")
        return False
    if args.global_rank is not None and args.global_rank < 0:
        logger.error(f"global_rank error! global_rank={args.global_rank}")
        return False
    if args.global_world_size is not None and args.global_world_size <= 0:
        logger.error(f"global_world_size error! global_world_size={args.global_world_size}")
        return False
    if args.npu_num_per_dp <= 0:
        logger.error(f"npu_num_per_dp error! npu_num_per_dp={args.npu_num_per_dp}")
        return False
    if args.npu_device_id < 0:
        logger.error(f"npu_device_id error! npu_device_id={args.npu_device_id}")
        return False
    if args.parent_pid < 0:
        logger.error(f"parent_pid error! parent_pid={args.parent_pid}")
        return False
    if not isinstance(args.shm_name_prefix, str) or not args.shm_name_prefix:
        logger.error(f"shm_name_prefix error! shm_name_prefix={args.shm_name_prefix}")
        return False
    if args.communication_type not in ["shared_meme", "http"]:
        logger.error(f"communication_type error! communication_type={args.communication_type}")
        return False
    if hasattr(args, 'layerwise_disaggregated'):
        if args.layerwise_disaggregated not in ["false", "true"]:
            logger.error(
                f"[layerwiseDisaggregated] layerwise_disaggregated error! "
                f"layerwise_disaggregated={args.layerwise_disaggregated}"
            )
            return False
        if args.layerwise_disaggregated_role_type not in ["", "master", "slave"]:
            logger.error(
                f"[layerwiseDisaggregated] role_type error! "
                f"role_type={args.layerwise_disaggregated_role_type}"
            )
            return False
    return True


def main() -> int:
    # 解析命令行参数
    config = parse_from_cmd()

    # 配置检查
    if not check_config(config):
        return -1

    # 启动自适应GC
    AdaptiveGarbageCollector.get_instance().start()  # monitor every 1 second
    GCMonitor.get_instance()

    # 启动模型
    request_listener: RequestListener = RequestListener.get_instance(config)

    # gc监控
    GCMonitor.get_instance()

    # 注册异常信号量处理
    register_signal(request_listener)

    if request_listener.start() is False:
        logger.error("request listener cannot be launched.")
        return -1

    # 停止自适应GC
    AdaptiveGarbageCollector.get_instance().stop()

    # 程序退出
    logger.info("agent stop!")

    # 使用参数
    return 0


def register_signal(request_listener: RequestListener):
    def handler_signal(signum, frame):
        logger.info(f"Python process get signal:{signum}")
        if signum == signal.SIGTERM:
            request_listener.stop()

    # kill命令
    signal.signal(signal.SIGTERM, handler_signal)
    signal.signal(signal.SIGINT, handler_signal)

if __name__ == "__main__":
    sys.exit(main())