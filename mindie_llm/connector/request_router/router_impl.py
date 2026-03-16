# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import math
import os
import threading
import copy
import numpy as np

from mindie_llm.connector.common import send_model_execute_response, send_transfer_response, send_command_response
from mindie_llm.connector.common.response_builder import ExecuteResponseBuilder
from mindie_llm.connector.common.input_metadata_builder import (
    convert_execute_model_request_to_input_metadata_composite,
    convert_pull_kv_request_to_input_metadata_composite,
    get_attribute_info,
    convert_bytes_to_list,
    make_dummy_input_metadata,
    ConvertPara,
    make_dummy_input_metadata_dmi_decoder,
)
from mindie_llm.connector.common.model_execute_data_pb2 import (
    ExecuteRequest,
    ExecuteType,
    ForwardType,
    PDErrorCode,
    LoraOperationType
)
from mindie_llm.connector.common.model_execute_data_pb2 import ExecuteResponse, PDLinkResponse, LoraOperationResponse

from mindie_llm.connector.common.input_metadata_composite import InputMetadataComposite
from mindie_llm.model_wrapper.utils.config import DmiConfig
from mindie_llm.model_wrapper.utils.error import ModelWrapperErrorCode
from mindie_llm.model_wrapper.utils.metrics import FileMetrics
from mindie_llm.model_wrapper.utils.npu_compile import set_npu_compile_mode
from mindie_llm.text_generator.generator import Generator
from mindie_llm.utils.status import MindieLlmStatusCode

from mindie_llm.utils.log.logging import logger
from mindie_llm.utils.log.logging_base import HandlerType
from mindie_llm.utils.prof.profiler import span_start, span_end, span_req, span_attr

from mindie_llm.utils.layerwise.request_metadata import lwd_metadata_manager
from mindie_llm.utils.layerwise.input_metadata import EdgeCloudInputMetadata, pd_exec_matadata_instance
from mindie_llm.text_generator.utils.generation_output import GenerationOutput
from mindie_llm.text_generator.utils.config import ResponseConfig
from mindie_llm.utils.log.error_code import ErrorCode, ErrorCodeException

NON_KVCACHE_TOKEN_NUM = 1
SRC_BLOCK_TABLE_KEY = "src_block_tables"
DST_BLOCK_TABLE_KEY = "dst_block_tables"
REQ_INDEX = "request_index"
MAX_SEQUENCE_IDS_FOR_CPP = 120  # 超过此数量使用C++构建响应

ERROR_CODE_TO_FINISH_REASON = {
    ErrorCode.TEXT_GENERATOR_OUT_OF_MEMORY: ResponseConfig.EXCEPTION_OOM,
}


def _print_component_error_log(e: BaseException):
    """
    请自动收集底层异常日志信息，并追加该方法，提升定位效率
    :param e: Exception
    :raise: RuntimeError
    """
    if "ACL stream synchronize failed" in str(e):
        logger.exception(
            """[HCCL]\t>>>HCCL execute error, please first contact related experts to locate and solve,
                         Exception: %s""",
            e,
        )
        raise RuntimeError(f"HCCL execute error, Exception: {e}") from e

    if "The Inner error is reported as above. The process exits for this inner error" in str(e):
        logger.exception(
            """[CANN]\t>>>CANN execute error, please first contact related experts to locate and solve,
                         Exception: %s""",
            e,
        )
        raise RuntimeError(f"CANN execute error, Exception: {e}") from e


class RouterImpl:
    __slots__ = (
        "config",
        "max_seq_len",
        "rank",
        "local_rank",
        "tp_size",
        "cp_size",
        "dp_size",
        "dp_rank_id",
        "generator",
        "block_size",
        "total_time",
        "is_mix_model",
        "has_inited",
        "metrics",
        "empty_batch_task_id",
        "lock",
        "block_id_for_empty_req",
        "is_inference_pause",
        "layerwise_disaggregated",
    )

    def __init__(self):
        self.config = None
        self.max_seq_len = None
        self.rank = None
        self.local_rank = None
        self.tp_size = 1
        self.cp_size = 1
        self.dp_size = 1
        self.dp_rank_id = 0
        self.generator = None
        self.block_size = None
        self.total_time = 0
        self.is_mix_model = False
        self.has_inited = False
        self.metrics = FileMetrics()
        self.empty_batch_task_id = -10
        self.lock = threading.Lock()
        self.block_id_for_empty_req = -1
        self.is_inference_pause = False
        self.layerwise_disaggregated = False

    @staticmethod
    def check_output(generate_output):
        if generate_output.token_ids is None or generate_output.eos_info is None:
            logger.error("[MIE04E13030A] Token_ids or eos_info is none")
            raise ValueError("Token_ids or eos_info is none")
        if generate_output.token_ids.shape is None or generate_output.eos_info.shape is None:
            logger.error("[MIE04E13030A] Token_ids or eos_info shape is none")
            raise ValueError("Token_ids or eos_info shape is none")
        if generate_output.token_ids.shape[0] == 0 or generate_output.eos_info.shape[0] == 0:
            logger.error("[MIE04E13030A] Token_ids's or eos_info's shape[0] must be not 0")
            raise ValueError("Token_ids's or eos_info's shape[0] must be not 0")
        if not np.all((generate_output.token_ids >= -1) & (generate_output.token_ids < 2 ** 32)) or not np.all(
                (generate_output.eos_info >= 0) & (generate_output.eos_info < 2 ** 32)
        ):
            logger.error("[MIE04E13030A] Token_ids or eos_info data must be uint32")
            raise ValueError("Token_ids or eos_info data must be uint32")

    @staticmethod
    def _get_id_to_block_table_map(**kwargs):
        start_src = 0
        start_dst = 0

        id_to_block_table_map = kwargs.get("id_to_block_table_map", None)
        segment_len = kwargs.get("segment_len", None)
        src_block_table = kwargs.get("src_block_table", None)
        dst_block_table = kwargs.get("dst_block_table", None)
        inst_ids = kwargs.get("inst_ids", None)
        index = kwargs.get("index", None)

        for inst_id in inst_ids:
            valid_block_num = 0
            for idx in range(segment_len):
                if src_block_table[start_src + idx] != -1:
                    valid_block_num += 1
                else:
                    break
            id_to_block_table_map[inst_id][SRC_BLOCK_TABLE_KEY].extend(
                src_block_table[start_src: start_src + valid_block_num]
            )
            id_to_block_table_map[inst_id][DST_BLOCK_TABLE_KEY].extend(
                dst_block_table[start_dst: start_dst + valid_block_num]
            )
            id_to_block_table_map[inst_id][REQ_INDEX].append(index)
            start_src += segment_len
            start_dst += valid_block_num
        
    def initialize(self, model_config: DmiConfig) -> dict:
        self.config = model_config
        self.max_seq_len = model_config.max_seq_len
        self.rank = model_config.rank
        self.local_rank = model_config.local_rank
        self.tp_size = model_config.tp_size
        self.dp_size = model_config.dp_size
        self.dp_rank_id = (self.rank // (self.cp_size * self.tp_size)) % self.dp_size
        logger.info("global rank id %s get model config: %s", self.rank, model_config.model_config)
        self.generator = Generator(model_config={**model_config.model_config})
        self.config.enable_mtp = self.generator.enable_mtp
        self.is_mix_model = self.generator.is_mix_model
        self.block_size = model_config.cache_block_size
        logger.info(">>>global rank:%s done ibis manager to device", self.rank)

        if hasattr(model_config, 'layerwise_disaggregated'):
            if model_config.layerwise_disaggregated is not None and model_config.layerwise_disaggregated == "true":
                self.layerwise_disaggregated = True

        set_npu_compile_mode()

        if model_config.max_seq_len > self.generator.max_position_embeddings:
            logger.warning(
                """[MIE04E13030A] >>>global rank:%s max_seq_len(%s)
                           is bigger than max_position_embeddings(%s)
                           from model(%s).
                           The accuracy of the inference result may be affected.""",
                self.rank,
                model_config.max_seq_len,
                self.generator.max_position_embeddings,
                model_config.model_weight_path,
            )
        
        num_npu_blocks = self.generator.kvcache_settings.num_npu_blocks
        if hasattr(self.generator.model_wrapper, "mapping") and self.generator.model_wrapper.mapping.has_dp():
            num_npu_blocks = num_npu_blocks - 1
        # 最后一个block 预留为陪跑使用
        self.block_id_for_empty_req = num_npu_blocks - 1
        kv_desc = [
            {
                "npuBlockNum": int(num_npu_blocks - 1),
                "blockSize": self.block_size,
                "compressionRatio": 1,
                "cacheType": 0,
            }
        ]
        initialize_result = {
            "status": "ok",
            "cpuBlockNum": str(self.generator.kvcache_settings.num_cpu_blocks),
            "maxPositionEmbeddings": str(self.generator.max_position_embeddings),
            "kvCacheDescs": kv_desc,
        }
        initialize_result["memPoolId"] = "-1"

        logger.info("model init success, parent pid=%s, pid=%s, device_id=%s, global_rank_id=%s, local_rank_id=%s",
                    os.getppid(), os.getpid(), self.config.npu_device_id, self.rank, self.config.local_rank)
        logger.info(">>>global rank:%s: return initialize success result: %s", self.rank, initialize_result)

        return initialize_result

    def execute(self, execute_request: ExecuteRequest):
        forward_type = execute_request.execute_model_request.forward_type
        if forward_type == ForwardType.DECODE:
            self._generate(execute_request, is_prefill=False, is_mix=False)
        elif forward_type == ForwardType.PREFILL:
            self._generate(execute_request, is_prefill=True, is_mix=False)
        elif forward_type == ForwardType.MIXED:
            self._mix(execute_request)
        elif forward_type == ForwardType.DUMMY:
            self._execute_empty_batch(execute_request)
        else:
            logger.error(
                """[MIE04E13030A] [Model]\t>>>
                Unknown forward_type: %s""",
                forward_type,
            )

    def seq_ctrl(self, execute_request: ExecuteRequest):
        logger.debug("[Model]\t>>> rank-%s enter seqctrl", self.rank)

        seq_id_tensor = [seq_id_to_clear for seq_id_to_clear in execute_request.text_generator_cleanup_request.seq_ids]
        if len(seq_id_tensor):
            seq_id = np.array(seq_id_tensor, dtype=np.int64, copy=False)
        else:
            raise ValueError("Failed to get 'SEQ_IDS_TO_CLEAR' value!")

        logger.debug(
            "[Model]\t>>> global rank-%s seqctrl method clear cache for seqId=%s, dp_rank_id=%s",
            self.rank,
            seq_id,
            self.dp_rank_id,
        )
        self.generator.clear_cache(seq_id)
        if self.layerwise_disaggregated and execute_request.execute_type == ExecuteType.TEXT_GENERATOR_CLEANUP:
            self.generator.plugin.set_clean_sequence_ids(execute_request.text_generator_cleanup_request.seq_ids)

    def transfer_data(self, execute_request: ExecuteRequest):
        input_metadata_composite: InputMetadataComposite = convert_pull_kv_request_to_input_metadata_composite(
            execute_request.pull_kv_request, self.generator.kvcache_settings.num_npu_blocks, self.block_size,
            self.config)
        self._get_pull_kv_items(execute_request, input_metadata_composite)

        prof = span_start("PullKVCache", domain="PullKVCache")
        span_req(prof, input_metadata_composite.input_metadata.batch_request_ids)
        span_attr(prof, "rank", self.rank)

        ret, failed_p_id = self.generator.pull_kv(
            input_metadata_composite.input_metadata, input_metadata_composite.pull_kv_items
        )

        span_end(prof)

        failed_p_id_set = set()
        if ret != MindieLlmStatusCode.SUCCESS:
            logger.error(
                "global rank-%s cluster-%s pull kv cache failed! pull_kv_items: %s, request_ids: %s",
                self.rank,
                failed_p_id,
                input_metadata_composite.pull_kv_items,
                input_metadata_composite.input_metadata.batch_request_ids,
            )
            failed_p_id_set.add(failed_p_id)

        # responses中存放request_id:result_code
        responses = dict()
        result_code = ModelWrapperErrorCode.PD_PULL_KV_ERROR if failed_p_id_set else ModelWrapperErrorCode.SUCCESS
        for pull_kv_info in execute_request.pull_kv_request.pull_kv_infos:
            request_id = pull_kv_info.seq_group_metadata.request_id
            # Caller uses errorCode in responses for actual result.
            responses[str(request_id)] = result_code

        # Return success so executor can release kv cache.
        result_code = ModelWrapperErrorCode.SUCCESS
        proto = ExecuteResponseBuilder.build_from_transfer_result(result_code.value, responses)
        send_transfer_response(proto)

    def pd_role(self, execute_request):
        logger.info("[Model]\t>>>global rank-%s enter process pdRole", self.rank)
        self.config.set_pd_link_info(get_attribute_info(execute_request.pd_link_request))

        logger.info("[Config]\t>>> start to process DMI link scenario.")
        # unlink
        logger.info(f"[Config] rank: {self.rank} Destroy all clusters kvcache link start...")
        for unlink_inst_cluster in set(self.config.remote_unlink_cluster_id):
            for unlink_cluster_item in self.config.remote_unlink_cluster_id[unlink_inst_cluster]:
                try:
                    logger.info(
                        f"[Config]\t>>> rank: {self.rank} " f"Destroy cluster id:{unlink_cluster_item} link start..."
                    )
                    self.generator.unlink(unlink_cluster_item)
                    logger.info(
                        f"[Config]\t>>> rank: {self.rank} " f"Destroy cluster id:{unlink_cluster_item} link finish..."
                    )
                except Exception as e:
                    _print_component_error_log(e)
                    logger.error("[Model]\t>>> process DMI unlink failed.")
                    logger.exception(f"[Model]\t>>> Exception:{e}")
                    return send_transfer_response(
                        ExecuteResponse(msg_type=ExecuteType.PD_LINK,
                                        status=ModelWrapperErrorCode.PD_UNLINK_ERROR.value,
                                        pd_link_response=PDLinkResponse())
                    )

        logger.info(f"[Config]\t>>> rank: {self.rank} Destroy all clusters kvcache link finish...")

        if self.config.need_switch:
            self.generator.switch_role(self.config.role)

        # link
        logger.info(f"[Config]\t>>> rank: {self.rank} Create all clusters kvcache links start...")
        failed_list = self.generator.link(
            remote_cluster_ids=self.config.remote_link_cluster_id,
            remote_physical_device_ids=self.config.remote_link_device_physical_id,
            remote_device_ips=self.config.remote_link_device_ips,
            host_ips=self.config.remote_link_host_ip,
            remote_super_device_ids=self.config.remote_super_device_id if self.config.remote_super_device_id else None,
            remote_super_pod_ids=self.config.remote_super_pod_id if self.config.remote_super_pod_id else None,
        )
        logger.debug(f"[Config]\t>>> rank: {self.rank} Create all clusters kvcache links finish...")

        if len(failed_list) == 0:
            # 处理成功，直接返回
            return send_transfer_response(
                ExecuteResponse(msg_type=ExecuteType.PD_LINK, pd_link_response=PDLinkResponse()))

        device_id_dict = {}
        for instance_id in self.config.remote_link_cluster_id.keys():
            id_unit = zip(
                self.config.remote_link_device_ips[instance_id], self.config.remote_link_cluster_id[instance_id]
            )
            device_id_dict.update({device_ip: cluster_id for device_ip, cluster_id in id_unit})
        link_fail_ip_list = []
        for device_ip, status in failed_list:
            if device_ip not in device_id_dict:
                logger.error(f"""[Config]\t>>> unknown device_ip '{device_ip}' in failed list.""")
                continue
            cluster_id = device_id_dict[device_ip]
            link_fail_ip_list.append([cluster_id, status.value])
            logger.error(f"""[Config]\t>>> global rank-{self.rank} cluster_id: {cluster_id} link fail.""")

        logger.info("[Model]\t>>>global rank-%s the total failed link server ip : %s", self.rank, link_fail_ip_list)

        proto_response = ExecuteResponse(msg_type=ExecuteType.PD_LINK, status=ModelWrapperErrorCode.PD_LINK_ERROR.value,
                                         pd_link_response=PDLinkResponse())

        for link_fail_ip in link_fail_ip_list:
            proto_response.pd_link_response.failed_link_info.append(PDLinkResponse.FailedLinkInfo(
                cluster_id=str(link_fail_ip[0]),
                pd_error_code=PDErrorCode.PD_LINK_ERROR
            ))

        return send_transfer_response(proto_response)

    def process_lora_operation(self, execute_request):
        lora_operation_type = execute_request.lora_operation_request.lora_op_type
        lora_name = execute_request.lora_operation_request.lora_name
        lora_path = execute_request.lora_operation_request.lora_path
        if lora_operation_type == LoraOperationType.LOAD:
            ret = self.generator.load_lora(lora_name, lora_path)
        elif lora_operation_type == LoraOperationType.UNLOAD:
            ret = self.generator.unload_lora(lora_name)
        else:
            logger.error(
                """[MIE04E13030A] [LORA]\t>>>
                Unknown lora_operation_type: %s""",
                lora_operation_type,
            )
        proto_response = ExecuteResponse(msg_type=ExecuteType.LORA_OPERATION,
                                         status=ModelWrapperErrorCode.SUCCESS.value,
                                         lora_operation_response=LoraOperationResponse(
                                             lora_name=lora_name,
                                             lora_op_status=ret
                                         )
                                         )
        return send_command_response(proto_response)

    def recover_command_exec(self, execute_request):
        command = execute_request.recover_command_request.command
        logger.debug("[Model]\t>>> rank-%s execute recover command: %s", self.rank, command)
        ret_dict = self.generator.execute_recover_command(command)
        proto_response = ExecuteResponseBuilder.build_from_recover_command_result(ret_dict, command)
        send_command_response(proto_response)

    def finalize(self):
        self.metrics.output()
        send_model_execute_response(ExecuteResponse(msg_type=ExecuteType.MODEL_FINALIZE))

    def _execute_empty_batch(self, execute_request):
        lwd_exe_stage = lwd_metadata_manager.get_metadata() if self.layerwise_disaggregated else None
        dummy_input_metadata = make_dummy_input_metadata(
            execute_request, self.generator.kvcache_settings.num_npu_blocks, self.config, lwd_exe_stage
        )
        if self.config.infer_mode == "dmi" and self.config.role == "decoder":
            self.generator.input_metadata_queue.put(dummy_input_metadata)
            dummy_input_metadata = make_dummy_input_metadata_dmi_decoder(dummy_input_metadata,
                                                                         self.generator.kvcache_settings.num_npu_blocks,
                                                                         self.config)
        logger.info(
            f"execute empty dummy batch with execute_type: {execute_request.execute_type}, "
            f"forward_type: {execute_request.execute_model_request.forward_type}",
            extra={"handler_ids": HandlerType.TOKEN}
        )
        err_msg = ""
        try:
            self.generator.generate_token(dummy_input_metadata)
        except ErrorCodeException as e:
            logger.error(f'{e.error_code.name} fault happened when handling empty batch, '
                         f'will be reported to executor with error code: {e.error_code.value}.')
            err_msg = e.error_code.value
            proto = ExecuteResponseBuilder.build_from_err_msg(err_msg)
            logger.info(f"Send error response to rank {self.local_rank}, err_msg={err_msg}")
            send_model_execute_response(proto)
        except Exception as e:
            logger.error(f'Unknown exception when handling empty batch, error: {e}')
            raise e

        proto = ExecuteResponse(msg_type=execute_request.execute_type)
        send_model_execute_response(proto)

    def _mix(self, execute_request: ExecuteRequest):
        is_req_prefill = []
        for seq_group_metadata in execute_request.execute_model_request.seq_group_metadata_list:
            is_req_prefill.extend(seq_group_metadata.is_req_prefill)
        is_prefill = True in is_req_prefill
        is_mix = (True in is_req_prefill) and (False in is_req_prefill)
        self._generate(execute_request, is_prefill=is_prefill, is_mix=is_mix)

    def _generate(self, execute_request: ExecuteRequest, is_prefill, is_mix):
        convert_prof = span_start("GetInputMetadata", domain="ModelExecute")
        input_metadata_composite: InputMetadataComposite = None
        if not self.layerwise_disaggregated:
            input_metadata_composite = convert_execute_model_request_to_input_metadata_composite(
                execute_request.execute_model_request,
                self.generator.kvcache_settings.num_npu_blocks,
                self.block_size,
                ConvertPara(is_prefill=is_prefill, is_mix=is_mix),
                is_mix_model=self.is_mix_model,
                layerwise_disaggregated_exe_stage=None,
                config=self.config
            )
            logger.info(
                f"execute real batch with batch_size: {input_metadata_composite.input_metadata.batch_size}, "
                f"execute_type: {execute_request.execute_type}, "
                f"forward_type: {execute_request.execute_model_request.forward_type}",
                extra={"handler_ids": HandlerType.TOKEN}
            )
        else:
            layerwise_disaggregated_exe_stage = lwd_metadata_manager.get_metadata()
            # For P-last, D-last, and non-first blocks of cloud-side P tasks, directly
            # use the result from the previous computation.
            if EdgeCloudInputMetadata.have_input_metadata(layerwise_disaggregated_exe_stage):
                pd_exec_matadata = pd_exec_matadata_instance
                input_metadata_composite = copy.deepcopy(
                    pd_exec_matadata.get_input_metadata(layerwise_disaggregated_exe_stage.is_prefill, 
                                                        layerwise_disaggregated_exe_stage))
                # The exe_stage for P-end differs from that of P-begin, and similarly,
                # the exe_stage for D-end differs from that of D-begin; these must be replaced accordingly.
                lw_disag_exe_stage = layerwise_disaggregated_exe_stage
                input_metadata_composite.input_metadata.layerwise_disaggregated_exe_stage = lw_disag_exe_stage
            else:
                input_metadata_composite = convert_execute_model_request_to_input_metadata_composite(
                    execute_request.execute_model_request,
                    self.generator.kvcache_settings.num_npu_blocks,
                    self.block_size,
                    ConvertPara(is_prefill=is_prefill, is_mix=is_mix),
                    is_mix_model=self.is_mix_model,
                    layerwise_disaggregated_exe_stage=layerwise_disaggregated_exe_stage,
                    config=self.config
                )
                # For P-first, D-first, and first-block cloud-side P tasks, back up the computation result to provide
                # for subsequent P-last, D-last, and non-first-block cloud-side P tasks.
                if EdgeCloudInputMetadata.need_storage_input_metadata(layerwise_disaggregated_exe_stage):
                    pd_exec_matadata = pd_exec_matadata_instance
                    exe_stage = layerwise_disaggregated_exe_stage
                    if exe_stage.is_prefill and exe_stage.is_long_seq:
                        pd_exec_matadata.set_input_metadata(copy.deepcopy(input_metadata_composite),
                                                    layerwise_disaggregated_exe_stage.is_prefill)
                    else:
                        pd_exec_matadata.set_input_metadata(input_metadata_composite,
                                                    layerwise_disaggregated_exe_stage.is_prefill)
        span_end(convert_prof)
        if input_metadata_composite.block_copy:
            self.generator.copy_blocks(np.array(input_metadata_composite.block_copy))
        
        err_msg = ""
        try:
            generate_output = self._handle_requests(input_metadata_composite)
        except ErrorCodeException as e:
            logger.error(f'{e.error_code.name} fault happened when handling normal batch, '
                         f'will be reported to executor with error code: {e.error_code.value}.')
            err_msg = e.error_code.value
            sequence_ids = input_metadata_composite.input_metadata.all_sequence_ids
            parent_sequence_ids = input_metadata_composite.input_metadata.all_sequence_ids
            dim = len(sequence_ids)
            finish_reason = np.zeros((dim), dtype=np.int32) + \
                ERROR_CODE_TO_FINISH_REASON.get(e.error_code, ResponseConfig.EOS).value
            generate_output = GenerationOutput(
                sequence_ids=sequence_ids,
                parent_sequence_ids=parent_sequence_ids,
                group_indices=[(i, i + 1) for i in range(dim)],
                logprobs=np.zeros((dim, 1), dtype=np.float32),
                top_token_ids=np.zeros((dim, 1, 1), dtype=np.int64),
                top_logprobs=np.zeros((dim, 1, 1), dtype=np.float32),
                num_new_tokens=np.ones((dim,), dtype=np.int64),
                num_top_tokens=np.zeros((dim,), dtype=np.int64),
                cumulative_logprobs=np.zeros((dim,), dtype=np.float32),
                token_ids=np.zeros((dim, 1), dtype=np.int64),
                finish_reason=finish_reason,
                truncation_indices=np.zeros((dim), dtype=np.int_),
                current_token_indices=np.zeros((dim), dtype=np.int_)
            )
            generate_output.collate()
            logger.info(f"Send response with error msg to rank {self.local_rank}, err_msg={err_msg}")
            proto = ExecuteResponseBuilder.build_from_err_msg(err_msg)
            send_model_execute_response(proto)
        except Exception as e:
            logger.error(f'Unknown exception when handling normal batch, error: {e}')
            raise e

        # Only the first rank in each DP group will generate output.
        if generate_output is not None:
            if self.local_rank % self.tp_size == 0 or self.config.distributed_enable:
                prof = span_start("GenerateOutput", domain="ModelExecute")
                if len(generate_output.sequence_ids) > MAX_SEQUENCE_IDS_FOR_CPP:
                    if not self.layerwise_disaggregated:
                        proto_binary = ExecuteResponseBuilder.build_from_generate_output_use_cpp(generate_output)
                    else:
                        proto_binary = ExecuteResponseBuilder.lwd_build_from_generate_output_use_cpp(generate_output,
                            is_prefill)
                    send_model_execute_response(proto_binary, True)
                else:
                    proto = ExecuteResponseBuilder.build_from_generate_output(
                        generate_output, execute_request.execute_type)
                    if self.layerwise_disaggregated:
                        proto.execute_model_response.layerwise_is_prefill = is_prefill
                    send_model_execute_response(proto)
                span_end(prof)
            else:
                # 其他卡发送一个空
                proto = ExecuteResponseBuilder.build_from_generate_output(None, execute_request.execute_type)
                send_model_execute_response(proto)

    def _get_pull_kv_items(self, execute_request: ExecuteRequest, input_metadata_composite: InputMetadataComposite):
        """获取pull_kv_items"""
        p_ip_block_table_map = {}
        batch_dst_block_tables = []
        batch_src_block_tables = []
        batch_req_ids = []
        batch_p_ip_int = []

        for pull_kv_info in execute_request.pull_kv_request.pull_kv_infos:
            batch_dst_block_tables.append(np.frombuffer(pull_kv_info.dst_block_tables[0], dtype=np.int64).tolist())
            batch_src_block_tables.append(np.frombuffer(pull_kv_info.src_block_tables[0], dtype=np.int64).tolist())
            batch_req_ids.append(pull_kv_info.seq_group_metadata.request_id)
            batch_p_ip_int.append(pull_kv_info.cluster_id)

        for i, pull_kv_info in enumerate(execute_request.pull_kv_request.pull_kv_infos):
            seq_group_metadata = pull_kv_info.seq_group_metadata
            dst_block_tables = np.array(batch_dst_block_tables[i], copy=False)
            src_block_tables = np.array(batch_src_block_tables[i], copy=False)
            request_req_ids = batch_req_ids[i]
            request_seqs_len = convert_bytes_to_list(seq_group_metadata.prompt_lens)
            # compute req needed block num
            # 当前一个seqgrp下只有一个seq；而且beamserch不支持pd分离，所以request_seqs_len中只有一个数据
            # pd分离时，D节点收到的input ids里面包含prefill节点推理出来的1个token，所以减去这个token才是真实的prompt的长度
            req_block_num = (int(request_seqs_len[0]) - NON_KVCACHE_TOKEN_NUM + self.block_size - 1) // self.block_size
            segment_len = 0

            # SP的时候，需要P节点传给D节点的src block table打好padding
            if self.config.p_inst_enable_sp_cp:
                segment_len = math.ceil(req_block_num / (self.config.remote_sp_size * self.config.remote_cp_size))
                req_block_num = segment_len * self.config.remote_sp_size * self.config.remote_cp_size
                # 需要padding TBC
            src_block_table = src_block_tables[:req_block_num]
            dst_block_table = dst_block_tables[:req_block_num]

            logger.debug(
                """[Model]\t>>>global rank-%s req_id: %s
                src_block_table: %s, dst_block_table:%s""",
                self.rank,
                request_req_ids[0],
                src_block_table,
                dst_block_table,
            )

            # 获取P节点的IP信息
            p_ip_int = batch_p_ip_int[i]
            if self.dp_size == 1:
                # 单机PD分离和大EP场景获取p_ip_ints的索引方式不一致。单机使用pinstance id，大EP使用dpinstanceid
                # pull_kv_info.cluster_id里传进来的是dpinstanceid，单机PD分离场景没有dp，使用pinstanceid
                # 换算公式 dpinstanceid = pinstance X 10000 + dprank
                p_ip_ints = self.config.dp_inst_id_to_cluster_id[int(p_ip_int) // 10000]
            else:
                p_ip_ints = self.config.dp_inst_id_to_cluster_id[int(p_ip_int)]

            for p_ip_int in p_ip_ints:
                if p_ip_int not in p_ip_block_table_map.keys():
                    p_ip_block_table_map[p_ip_int] = {}
                    p_ip_block_table_map[p_ip_int][SRC_BLOCK_TABLE_KEY] = []
                    p_ip_block_table_map[p_ip_int][DST_BLOCK_TABLE_KEY] = []
                    p_ip_block_table_map[p_ip_int][REQ_INDEX] = []
            if self.config.p_inst_enable_sp_cp:
                RouterImpl._get_id_to_block_table_map(
                    id_to_block_table_map=p_ip_block_table_map,
                    segment_len=segment_len,
                    src_block_table=src_block_table,
                    dst_block_table=dst_block_table,
                    inst_ids=p_ip_ints,
                    index=i,
                )
            else:
                p_ip_block_table_map[p_ip_int][SRC_BLOCK_TABLE_KEY].extend(src_block_table)
                p_ip_block_table_map[p_ip_int][DST_BLOCK_TABLE_KEY].extend(dst_block_table)
                p_ip_block_table_map[p_ip_int][REQ_INDEX].append(i)

        pull_kv_items = []
        for p_ip_int, block_table in p_ip_block_table_map.items():
            if len(block_table[SRC_BLOCK_TABLE_KEY]) == 0:
                continue
            p_src_block_tables = [int(block) for block in block_table[SRC_BLOCK_TABLE_KEY]]
            d_src_block_tables = [int(block) for block in block_table[DST_BLOCK_TABLE_KEY]]
            pull_kv_items.append((int(p_ip_int), p_src_block_tables, d_src_block_tables))

        logger.info(
            "[Pull kv]\t>>>global rank-%s pull kv batch_size is %s, pull kv info is %s, size is %s, reqId is: %s",
            self.rank,
            len(execute_request.pull_kv_request.pull_kv_infos),
            pull_kv_items,
            len(pull_kv_items),
            input_metadata_composite.input_metadata.batch_request_ids,
        )
        input_metadata_composite.pull_kv_items = pull_kv_items

    def _prepare_kv_block(self, input_metadata_composite: InputMetadataComposite):
        if input_metadata_composite.block_op:
            self.generator.swap(input_metadata_composite.block_op)

    def _handle_requests(self, input_metadata_composite: InputMetadataComposite):
        # prepare kv blocks for request
        # do operations including preempt swap/recompute, load prefix cache from host
        self._prepare_kv_block(input_metadata_composite)

        metadata = input_metadata_composite.input_metadata

        generate_output = self.generator.generate_token(metadata)
        if generate_output is None:
            return None
        # 将数组中为0的元素都改成1
        generate_output.num_top_tokens[generate_output.num_top_tokens == 0] = 1
        if not self.is_inference_pause:
            RouterImpl.check_output(generate_output)
        return generate_output
