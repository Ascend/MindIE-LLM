/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include <memory>
#include "grpc_communication_mng.h"
#include "infer_instances.h"
#include "http_rest_resource.h"
#include "config_manager.h"
#include "endpoint_def.h"
#include "single_llm_decode_req_handler.h"
#include "single_req_infer_interface_base.h"
#include "single_req_triton_text_infer_interface.h"
#include "single_req_tgi_text_infer_interface.h"
#include "single_req_vllm_openai_infer_interface.h"
#include "single_req_vllm_openai_completions_infer_interface.h"
#include "single_req_openai_infer_interface.h"
#include "single_req_self_develop_infer_interface.h"
#include "single_req_triton_token_infer_interface.h"
#include "single_req_vllm_infer_interface.h"
#include "grpc_context.h"
#include "log.h"
#include "dmi_role.h"
#include "config_manager_impl.h"
#include "grpc_handler.h"

namespace mindie_llm {
void HandleDecodeRequest(const prefillAndDecodeCommunication::DecodeParameters &para,
    prefillAndDecodeCommunication::DecodeRequestResponse &response)
{
    static uint64_t receiveCnt = 0;
    auto prof = PROF(L2, Domain("Communication").Resource(para.reqid()).SpanStart("handleDecodeRequest"));
    PROF(prof.Attr("InstanceId", para.pinstanceid()));
    PROF(prof.NumArrayAttr("tokens", para.tokens().begin(), para.tokens().end()));
    PROF(prof.NumArrayAttr("firsttoken", para.firsttoken().begin(), para.firsttoken().end()));
    PROF(prof.ArrayAttr("outputnames", para.outputnames().begin(), para.outputnames().end(),
        [](decltype(prof)* pColl, decltype(para.outputnames().begin()) item) -> void {
            pColl->Attr("name", *item);
        }));

    PROF(prof.Attr("maxnewtoken", para.maxnewtoken()));
    PROF(prof.Attr("truncate", para.truncate()));
    PROF(prof.Attr("tools", para.tools()));
    PROF(prof.Attr("toolchoice", para.toolchoice()));
    PROF(prof.Attr("loraid", para.loraid()));
    PROF(prof.Attr("returnfulltext", para.returnfulltext()));
    PROF(prof.Attr("decoderinputdetails", para.decoderinputdetails()));
    PROF(prof.Attr("modelname", para.modelname()));
    PROF(prof.Attr("details", para.details()));
    PROF(prof.Attr("id", para.id()));
    PROF(prof.Attr("e2estarttime", para.e2estarttime()));
    if (para.blocktable_size() > 0) {
        PROF(prof.NumArrayAttr("blocktable", para.blocktable()[0].blockid().begin(),
                               para.blocktable()[0].blockid().end()));
    }
    PROF(prof.Attr("prevdecodeindex", para.prevdecodeindex()));
    PROF(prof.Attr("currentdecodeindex", para.currentdecodeindex()));
    PROF(prof.Attr("postsingletext", para.postsingletext()));
    if (para.inputid().has_value()) {
        PROF(prof.Attr("inputid", para.inputid().value()));
    }
    if (para.textinput().has_value()) {
        PROF(prof.Attr("textinput", para.textinput().value()));
    }
    PROF(prof.NumArrayAttr("dpinstanceids", para.dpinstanceids().begin(), para.dpinstanceids().end()));

    // pdRole flex是纯P节点时候不接收流量
    std::string pdRoleName = GetInferInstance()->GetPDRole();
    uint32_t maxFlexPPercentage = 100;
    if (pdRoleName == "flex" &&
        FlexPPercentageProcessor::GetInstance().GetPdRoleFlexPPercentage() == maxFlexPPercentage) {
        response.set_errormessage("failed to register decode request handler, pdRole flex p_percentage = 100.");
        response.set_isvaliddecodeparameters(false);
        return;
    }
    KvCacheInfo kvCacheInfo;
    
    kvCacheInfo.blockTable.resize(para.blocktable_size());
    for (int i = 0; i < para.blocktable_size(); ++i) {
        const auto &blocktable = para.blocktable(i);
        kvCacheInfo.blockTable[i].reserve(blocktable.blockid_size());
        for (int j = 0; j < blocktable.blockid_size(); ++j) {
            kvCacheInfo.blockTable[i].push_back(blocktable.blockid(j));
        }
    }
    
    for (int i = 0; i < para.dpinstanceids_size(); ++i) {
        kvCacheInfo.dpInstanceIds.push_back(para.dpinstanceids()[i]);
    }
    DmiServerInfo serverInfo(
        para.reqid(), para.pnodeaddr(), "", kvCacheInfo, InferReqType::REQ_DECODE);
    auto gctx = std::make_shared<GrpcContext>(serverInfo);
    gctx->SetDecodeParams(para);
    std::shared_ptr<RequestContext> context{nullptr};
    auto tmpDispatcher = atomic_load(&dResultDispatcher);
    std::shared_ptr<SingleLLMDecodeReqHandler> singleLLMDecodeReqHandler =
        std::make_shared<SingleLLMDecodeReqHandler>(context, tmpDispatcher, gctx);
    std::shared_ptr<SingleReqInferInterfaceBase> inferInterface{nullptr};
    switch (static_cast<MsgType>(para.msgtype())) {
        case MsgType::MSG_TYPE_TRITON:
            gctx->SetTritonTextInfo({para.id()});
            inferInterface = std::make_shared<SingleReqTritonTextInferInterface>(singleLLMDecodeReqHandler);
            break;
        case MsgType::MSG_TYPE_TRITON_TOKEN:
            gctx->SetTritonTextInfo({para.id()});
            inferInterface = std::make_shared<SingleReqTritonTokenInferInterface>(singleLLMDecodeReqHandler);
            break;
        case MsgType::MSG_TYPE_OPENAI:
            inferInterface = std::make_shared<SingleReqOpenAiInferInterface>(singleLLMDecodeReqHandler);
            break;
        case MsgType::MSG_TYPE_VLLM_OPENAI:
            inferInterface = std::make_shared<SingleReqVllmOpenAiInferInterface>(singleLLMDecodeReqHandler);
            break;
        case MsgType::MSG_TYPE_VLLM_OPENAI_COMP:
            inferInterface = std::make_shared<SingleReqVllmOpenAiCompletionsInferInterface>(singleLLMDecodeReqHandler);
            break;
        case MsgType::MSG_TYPE_INFER:
            inferInterface = std::make_shared<SingleReqSelfDevelopInferInterface>(singleLLMDecodeReqHandler);
            break;
        case MsgType::MSG_TYPE_VLLM:
            inferInterface = std::make_shared<SingleReqVllmInferInterface>(singleLLMDecodeReqHandler);
            break;
        case MsgType::MSG_TYPE_TGI:
            inferInterface = std::make_shared<SingleReqTgiTextInferInterface>(singleLLMDecodeReqHandler);
            break;
        case MsgType::MSG_TYPE_GENERAL_TGI:
            inferInterface = std::make_shared<SingleReqGeneralTgiTextInferInterface>(singleLLMDecodeReqHandler);
            break;
        default:
            response.set_errormessage("Unsupported message type: " + std::to_string(para.msgtype()));
            response.set_isvaliddecodeparameters(false);
            return;
    }
    response.set_errormessage("");
    response.set_isvaliddecodeparameters(true);
    if (inferInterface != nullptr) {
        inferInterface->SetDMIReComputeBuilder();
        inferInterface->DecodeProcess(response);
    } else {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            INIT_ERROR), "SingleReqInferInterfaceBase processImpl is null");
    }
};

void HandleKvRelease(const std::string &requestId)
{
    RequestIdNew reqId(requestId);
    static uint64_t releaseCnt = 0;

    Status status = GetInferInstance()->ControlRequest(reqId, OperationV2::RELEASE_KV);
    if (status.StatusCode() != Error::Code::OK) {
        ULOG_WARN(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_SPLITWISE,
            STATUS_WARNING), "Failed release request. requestId: " << requestId);
    } else {
        ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Release request. requestId: " << requestId);
    }
    return;
};

GrpcHandler& GrpcHandler::GetInstance()
{
    static GrpcHandler instance;
    return instance;
}

bool GrpcHandler::InitGrpcService()
{
    if (isReady_) {
        return true;
    }
    if (!GrpcCommunicationMng::GetInstance().Init(
        GetServerConfig().interCommTLSEnabled,
        GetServerConfig().ipAddress,
        std::to_string(GetServerConfig().interCommPort))) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            INIT_ERROR), "Failed to init grpc communication manager");
        return false;
    }
    isReady_.store(true);
    return true;
}
bool GrpcHandler::InitDmiBusiness()
{
    if (isReady_) {
        return true;
    }
    // 注册 decode 请求消息
    if (!GrpcCommunicationMng::GetInstance().RegisterDecodeRequestHandler(HandleDecodeRequest)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            ABNORMAL_TRANSMISSION_ERROR), "Failed to register decode request handler");
        return false;
    }
    // 注册 kv release 消息
    if (!GrpcCommunicationMng::GetInstance().RegisterKvReleaseHandler(HandleKvRelease)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            ABNORMAL_TRANSMISSION_ERROR), "Failed to register kv release handler");
        return false;
    }
    return true;
}

} // namespace mindie_llm
