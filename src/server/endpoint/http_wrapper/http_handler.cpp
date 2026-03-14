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
#define CPPHTTPLIB_OPENSSL_SUPPORT

#include <unistd.h>
#include <cstring>
#include <ctime>
#include <thread>
#include <chrono>
#include <unordered_map>
#include <nlohmann/json.hpp> // 需要安装 nlohmann/json 库
#include "llm_manager_v2.h"
#include "memory_utils.h"
#include "config_manager.h"
#include "string_utils.h"
#include "endpoint_def.h"
#include "http_wrapper.h"
#include "parse_protocol.h"
#include "single_req_triton_text_infer_interface.h"
#include "single_req_openai_infer_interface.h"
#include "single_req_vllm_openai_infer_interface.h"
#include "single_req_vllm_openai_completions_infer_interface.h"
#include "single_req_self_develop_infer_interface.h"
#include "single_req_vllm_infer_interface.h"
#include "single_req_tgi_text_infer_interface.h"
#include "single_req_triton_token_infer_interface.h"
#include "event_dispatcher.h"
#include "infer_instances.h"
#include "common_util.h"
#include "single_llm_pnd_req_handler.h"
#include "single_llm_prefill_req_handler.h"
#include "dresult_event_dispatcher.h"
#include "infer_tokenizer.h"
#include "dmi_role.h"
#include "prometheus_metrics.h"
#include "log.h"
#include "check_utils.h"
#include "config_manager_impl.h"
#include "safe_io.h"
#include "http_handler.h"

using namespace std;
using json = nlohmann::json;
using OrderedJson = nlohmann::ordered_json;
using namespace mindie_llm;

int64_t HttpHandler::startTime = 0;
std::mutex HttpHandler::dmiRoleMutex_;
static std::mutex g_dResMutex;
static constexpr int JUDGE_PROCESS_WAIT_TIME = 100;
constexpr int WAIT_TIME = 5; // 5s

namespace mindie_llm {
struct ExceptionInfo {
    int stateCode;
    const char *bodyParam;
};

const std::string MODELS_NAME_RE = "([\\w\\-.]+)";

static const std::unordered_map<PDRoleStatus, std::string> PD_STATUS_MAP = {
    {PDRoleStatus::READY, "RoleReady"},
    {PDRoleStatus::SWITCHING, "RoleSwitching"},
    {PDRoleStatus::UNKNOWN, "RoleUnknown"},
};

// codinator 长链接处理
std::shared_ptr<DResultEventDispatcher> dResultDispatcher = nullptr;
std::atomic<bool> keepAlive{false};
static void SendDResultKa()
{
    std::string ka = "ka:heartbeat";
    char separator = '\0';
    std::string msg = "";

    msg.append(ka).append(1, separator);

    std::string logMsg = "Send DResult KeepAlive message: " + msg;
    dResultDispatcher->SendEvent(msg, false, "KeepAliveMSG");
}

static void DResultKeepAlive()
{
    const auto interval = boost::posix_time::seconds(1);
    constexpr auto maxInterval = boost::chrono::seconds(5);
    boost::asio::io_context ioContext;
    boost::asio::deadline_timer timer(ioContext, interval);
    while (keepAlive.load()) {
        timer.expires_from_now(interval);
        timer.wait();
        {
            std::unique_lock<std::mutex> lk(g_dResMutex);
            if (!dResultDispatcher) {
                std::cout << "dResultDispatcher id null" << std::endl;
                break;
            }
            if (dResultDispatcher != nullptr && dResultDispatcher->GetIntervalFromPrevSend() > maxInterval) {
                SendDResultKa();
            }
        }
    }
    return;
}

static void HandleDResult(const ReqCtxPtr &reqCtx)
{
    std::unique_lock<std::mutex> lk(g_dResMutex);
    auto tmp = atomic_load(&dResultDispatcher);
    if (tmp != nullptr) {
        std::string msg = "";
        DResultWrapParam param {
            "new connection is to established, close the old one.",
            "close:",
            "0xffffffffffffffff",
            ""};
        DResultEventDispatcher::WrapChunkedDResponse(msg, param);
        tmp->SendEvent(msg, true, "CloseOldConnection");
    }
    auto disp = std::make_shared<DResultEventDispatcher>();
    if ((disp == nullptr) || (reqCtx == nullptr)) {
        return;
    }
    atomic_store(&dResultDispatcher, disp);
    auto &response = reqCtx->Res();
    response.set_chunked_content_provider("text/event-stream",
        [disp](size_t /* offset */, httplib::DataSink &sink) {
            disp->WaitEvent(&sink);
            return true;
        });
    return;
}

static void HandleGetInfoQuery(const std::shared_ptr<RequestContext> &context)
{
    std::string jsonStr = "";
    const ScheduleConfig &scheduleParam = GetScheduleConfig();
    const std::vector<ModelDeployConfig> &modelParam = GetModelDeployConfig();
    JsonParse::HandleGetInfo(scheduleParam, modelParam, jsonStr);

    HttpRestResource::ResponseJsonBody(context, -1, jsonStr);
}

// 处理 codinator 传来的请求的头
static InferReqType GetReqType(const std::string &typeStr, bool isFlexLocal)
{
    InferReqType reqType = InferReqType::REQ_STAND_INFER;
    if (typeStr == "prefill" && !isFlexLocal) {
        reqType = InferReqType::REQ_PREFILL;
    }
    return reqType;
}

static bool IsFlexLocalReq(const httplib::Request &request)
{
    std::string localIp = GetServerConfig().ipAddress;
    std::string interCommPort = std::to_string(GetServerConfig().interCommPort);
    std::string dTarget = request.get_header_value("d-target");
    if (dTarget.empty()) {
        return false;
    }
    std::vector<std::string> ipAndPort;
    mindie_llm::Split(dTarget, IP_PORT_DELIMITER, ipAndPort);
    if (ipAndPort.size() == 1) {
        return ipAndPort[0] == localIp;
    } else if (ipAndPort.size() == 2) { // should has ip and port 2 items
        return ipAndPort[0] == localIp && ipAndPort[1] == interCommPort;
    } else {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SERVER_REQUEST, CHECK_ERROR),
            "header dTarget " << dTarget << " is invalid");
        return false;
    }
}

static std::shared_ptr<HttpReqHeadersOption> MakeHttpHeadersOpt(const httplib::Request &request)
{
    auto option = std::make_shared<HttpReqHeadersOption>();
    if (option == nullptr) {
        return nullptr;
    }
    option->isFlexLocal = IsFlexLocalReq(request);
    option->reqType = GetReqType(request.get_header_value("req-type"), option->isFlexLocal);
    option->isReCompute = request.get_header_value("is-recompute") == "true" ? true : false;

    return option;
}

static Status GetRemainBlockNum(uint64_t &remainBlocks, std::map<uint32_t, uint64_t> *dpRemainBlocksOut = nullptr)
{
    uint64_t remainPrefillSlots = 0;
    uint64_t remainPrefillTokens = 0;
    std::map<uint32_t, uint64_t> dpRemainBlocksDummy;
    std::map<uint32_t, uint64_t> &dpRemainBlocks = dpRemainBlocksOut ? *dpRemainBlocksOut : dpRemainBlocksDummy;
    Status statusBlock = GetInferInstance()->GetRequestBlockQuotas(remainBlocks, remainPrefillSlots,
                                                                   remainPrefillTokens, dpRemainBlocks);
    if (!statusBlock.IsOk()) {
        ULOG_WARN(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_PREFIX_CACHE, STATUS_WARNING),
                  "Failed to get request block quotas");
    }
    return statusBlock;
}

static Status GetAvalSlotNum(uint64_t &availableSlots)
{
    uint64_t processReq = 0;
    Status statusFreeSlots = GetInferInstance()->GetProcessingRequest(processReq);
    if (!statusFreeSlots.IsOk()) {
        ULOG_WARN(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_PREFIX_CACHE,
            STATUS_WARNING), "Failed to get free slots");
        return statusFreeSlots;
    }

    auto &scheduleParam = GetScheduleConfig();
    availableSlots = (scheduleParam.maxBatchSize < processReq) ? 0 : scheduleParam.maxBatchSize - processReq;
    return Status(Error::Code::OK);
}

template <typename BuildInterfaceFn>
void DispatchInfer(ReqCtxPtr &reqCtx, const std::shared_ptr<HttpReqHeadersOption> &opt, MsgType prefillMsgType,
                   BuildInterfaceFn &&buildInterface, std::optional<RequestIdNew> stopRequestId = std::nullopt)
{
    std::shared_ptr<SingleReqInferInterfaceBase> singleReqInferInterface = nullptr;

    if (opt->reqType == InferReqType::REQ_STAND_INFER) {
        // for PD-mixed deployment
        std::shared_ptr<SingleLLMPnDReqHandler> singleLLMPnDReqHandler =
            std::make_shared<SingleLLMPnDReqHandler>(reqCtx, opt->isFlexLocal);
        singleReqInferInterface = buildInterface(singleLLMPnDReqHandler);
        RequestIdNew stopId = stopRequestId.has_value() ? *stopRequestId : singleReqInferInterface->GetRequestId();
        reqCtx->SetStopInferFunction(
            [stopId]() { (void)GetInferInstance()->ControlRequest(stopId, OperationV2::STOP); });
    } else if (opt->reqType == InferReqType::REQ_PREFILL) {
        // for PD disaggregation, where P accepts HTTP requests while D accetps gRPC requests from P
        std::shared_ptr<SingleLLMPrefillReqHandler> singleLLMPrefillReqHandler =
            std::make_shared<SingleLLMPrefillReqHandler>(reqCtx, prefillMsgType, opt->isReCompute);
        singleReqInferInterface = buildInterface(singleLLMPrefillReqHandler);
    } else {
        std::string errorResponseStr = "Request type does not match type InferReqType.";
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, CHECK_ERROR),
                   errorResponseStr);
        HttpRestResource::ResponseJsonBody(
            reqCtx, httplib::StatusCode::UnprocessableContent_422,
            HttpRestResource::WrapperJson(errorResponseStr,
                                          g_exceptionInfo.at(httplib::StatusCode::UnprocessableContent_422)));
        return;
    }
    singleReqInferInterface->Process();
}

void HttpHandler::SetJsonObj(ordered_json &jsonObj)
{
    uint64_t availSlotsNum = 0;
    uint64_t availBlockNum = 0;
    uint64_t waitingRequestNum = 0;
    uint64_t runningRequestNum = 0;
    uint64_t swappedRequestNum = 0;
    uint64_t freeNpuBlockNums = 0;
    uint64_t freeCpuBlockNums = 0;
    uint64_t totalNpuBlockNums = 0;
    uint64_t totalCpuBlockNums = 0;

    GetInferInstance()->GetWaitingRequest(waitingRequestNum);
    GetInferInstance()->GetRunningRequest(runningRequestNum);
    GetInferInstance()->GetSwappedRequest(swappedRequestNum);
    GetInferInstance()->GetCacheBlockNums(freeNpuBlockNums, freeCpuBlockNums, totalNpuBlockNums, totalCpuBlockNums);

    jsonObj["resource"]["availSlotsNum"] = GetAvalSlotNum(availSlotsNum).IsOk() ? availSlotsNum : 0;
    jsonObj["resource"]["availBlockNum"] = GetRemainBlockNum(availBlockNum).IsOk() ? availBlockNum : 0;
    jsonObj["resource"]["waitingRequestNum"] = waitingRequestNum;
    jsonObj["resource"]["runningRequestNum"] = runningRequestNum;
    jsonObj["resource"]["swappedRequestNum"] = swappedRequestNum;
    jsonObj["resource"]["freeNpuBlockNums"] = freeNpuBlockNums;
    jsonObj["resource"]["freeCpuBlockNums"] = freeCpuBlockNums;
    jsonObj["resource"]["totalNpuBlockNums"] = totalNpuBlockNums;
    jsonObj["resource"]["totalCpuBlockNums"] = totalCpuBlockNums;
    return;
}

// 更新动态状态检测接口
void HttpHandler::HandleStatusV1(const ReqCtxPtr &ctx)
{
    std::string status = "RoleReady";

    if (IsDMI()) {
        PDRoleStatus roleStatus = GetInferInstance()->GetPDRoleStatus();
        auto it = PD_STATUS_MAP.find(roleStatus);
        if (it != PD_STATUS_MAP.end()) {
            status = it->second;
        } else {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                CHECK_ERROR), "Role status is invalid.");
            HttpRestResource::ResponseJsonBody(ctx, httplib::StatusCode::InternalServerError_500,
                HttpRestResource::WrapperJson("Role status is invalid.",
                g_exceptionInfo.at(httplib::StatusCode::InternalServerError_500)));
            return;
        }
    }
    std::string currRole = GetInferInstance()->GetPDRole();

    ordered_json jsonObj;
    jsonObj["service"]["roleStatus"] = status;
    jsonObj["service"]["currentRole"] = currRole;
    if (currRole == "flex") {
        jsonObj["service"]["p_percentage"] = FlexPPercentageProcessor::GetInstance().GetPdRoleFlexPPercentage();
    }
    SetJsonObj(jsonObj);
    ordered_json linkStatusJson = ordered_json::array();

    for (const auto& pair : DmiRole::GetInstance()->GetRemoteNodeLinkStatus()) {
        if (pair.second.second) {
            ordered_json peersJson;
            peersJson["target"] = pair.first; // instance id
            peersJson["link"] = pair.second.first; // status info
            linkStatusJson.push_back(peersJson);
        }
    }
    jsonObj["linkStatus"]["peers"] = linkStatusJson;

    HttpRestResource::ResponseJsonBody(ctx, httplib::StatusCode::OK_200, jsonObj.dump());
}


bool ValidatePDRoleStatus(const ReqCtxPtr &ctx, std::string& statusString)
{
    PDRoleStatus roleStatus = GetInferInstance()->GetPDRoleStatus();
    auto it = PD_STATUS_MAP.find(roleStatus);
    if (it != PD_STATUS_MAP.end()) {
        statusString = it->second;
        return true;
    }
    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
        CHECK_ERROR), "Role status is invalid.");
    HttpRestResource::ResponseJsonBody(ctx, httplib::StatusCode::InternalServerError_500,
        HttpRestResource::WrapperJson("Role status is invalid.",
        g_exceptionInfo.at(httplib::StatusCode::InternalServerError_500)));
    return false;
}

// 更新动态状态检测借口
void HttpHandler::HandleStatusV2(const ReqCtxPtr &ctx)
{
    try {
        std::string status = "RoleReady";
        // When in pd separate case, check role status first
        if (IsDMI() && !ValidatePDRoleStatus(ctx, status)) {
            return;
        }
        std::string currRole = GetInferInstance()->GetPDRole();
        ordered_json jsonObj;
        uint64_t availSlotsNum = 0;
        uint64_t availBlockNum = 0;
        std::map<uint32_t, uint64_t> dpRemainBlocks;
        jsonObj["service"]["roleStatus"] = status;
        jsonObj["service"]["currentRole"] = currRole;
        SetJsonObj(jsonObj);
        jsonObj["resource"]["totalAvailNpuSlotsNum"] = GetAvalSlotNum(availSlotsNum).IsOk() ? availSlotsNum : 0;
        jsonObj["resource"]["totalAvailNpuBlockNum"] =
            GetRemainBlockNum(availBlockNum, &dpRemainBlocks).IsOk() ? availBlockNum : 0;
        jsonObj["resource"]["dpInstMap"] = dpRemainBlocks;
        if (dpRemainBlocks.empty()) {
            jsonObj["resource"]["maxAvailNpuBlockNum"] = 0;
        } else {
            jsonObj["resource"]["maxAvailNpuBlockNum"] = std::max_element(dpRemainBlocks.begin(), dpRemainBlocks.end(),
            [](const auto& a, const auto& b) {
                return a.second < b.second;
            })->second;
        }
        ordered_json linkStatusJson = ordered_json::array();
        auto remoteLinkStatusMap = DmiRole::GetInstance()->GetRemoteNodeLinkStatusV2();

        for (const auto& pair : remoteLinkStatusMap) {
            // pair <instanceId, <statusFlag, statusString>>
            if (pair.second.second) {
                ordered_json peersJson;
                peersJson["target"] = pair.first; // instance id
                peersJson["link"] = pair.second.first; // status info
                linkStatusJson.push_back(peersJson);
            }
        }
        jsonObj["linkStatus"]["peers"] = linkStatusJson;
        HttpRestResource::ResponseJsonBody(ctx, httplib::StatusCode::OK_200, jsonObj.dump());
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            CHECK_ERROR), "Handle StatusV2 Failed");
    }
}

/* 收到处理请求后，不能立即发送header，因为把isFinished设置false时，再发送失败请求会导致http状态机错误，失败请求无法发送
case 1:成功发送时，先发header,再发data
case 2:异常发送时，不发header,直接回复异常消息
case 3:流式返回时，中间异常不能发送失败请求，而是直接正常返回空字符结束
*/
bool CheckWaitTime(const std::string &waitTime, uint32_t &timeNum, ReqCtxPtr &requestContext)
{
    if (waitTime.empty()) {
        timeNum = SIMULATE_CV_WAIT_TIME;
        return true;
    }
    uint32_t maxLength = 3;
    std::string errorReason {};
    std::string numStr = waitTime.substr(1);
    if (numStr.size() > maxLength) {
        errorReason = "Max wait time is " + std::to_string(CV_WAIT_TIME) + ", input is invalid or too long";
        HttpRestResource::ResponseJsonBody(requestContext, httplib::StatusCode::NotFound_404,
            HttpRestResource::WrapperJson(errorReason, g_exceptionInfo.at(httplib::StatusCode::NotFound_404)));
        return false;
    }
    unsigned long val = 0;
    const uint32_t decimal = 10;
    // Input is limited to "(-[0-9]+)", and size is no bigger than 3.
    try {
        val = std::stoul(numStr, nullptr, decimal);
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            CHECK_ERROR), "Invalid input waitTime");
        errorReason = "Max wait time is " + std::to_string(CV_WAIT_TIME) + ", input is invalid or too long";
        HttpRestResource::ResponseJsonBody(requestContext, httplib::StatusCode::NotFound_404,
            HttpRestResource::WrapperJson(errorReason, g_exceptionInfo.at(httplib::StatusCode::NotFound_404)));
        return false;
    }
    uint32_t fval = static_cast<uint32_t>(val);
    if (fval > CV_WAIT_TIME || fval == 0) {
        errorReason = "Wait time should be in range of [1, " + std::to_string(CV_WAIT_TIME) + "], input is not valid";
        HttpRestResource::ResponseJsonBody(requestContext, httplib::StatusCode::NotFound_404,
            HttpRestResource::WrapperJson(errorReason, g_exceptionInfo.at(httplib::StatusCode::NotFound_404)));
        return false;
    }
    timeNum = fval;
    return true;
}

bool CheckHealthAndStop(ReqCtxPtr &context)
{
    if (StopServiceOption::stopServiceFlag.load()) {
        HttpRestResource::ResponseJsonBody(context, httplib::StatusCode::ServiceUnavailable_503,
            HttpRestResource::WrapperJson("The service has been stopped.",
            g_exceptionInfo.at(httplib::StatusCode::ServiceUnavailable_503)));
        return true;
    }
    if (!HealthManager::GetHealth()) {
        HttpRestResource::ResponseNobody(context, httplib::StatusCode::ServiceUnavailable_503);
        return true;
    }
    return false;
}

bool HttpHandler::HandlePostGenerate(ReqCtxPtr &reqCtx)
{
    uint16_t inferType = MSG_TYPE_INVALID;
    auto code = JsonParse::GetInferTypeFromJsonStr(reqCtx->MsgBody(), inferType);
    if (code != EP_OK || (inferType != MSG_TYPE_TGI && inferType != MSG_TYPE_VLLM)) {
        std::string jsonStr(R"delimiter({"Error": "`inputs` or `prompt`)delimiter" +
            std::string(" must be necessary and data type must be string. Additionally, the ") +
            R"delimiter(request body must be valid json", "error_type": "validation"})delimiter");
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            CHECK_ERROR), "Validation failed");
        HttpRestResource::ResponseJsonBody(reqCtx, httplib::StatusCode::UnprocessableContent_422,
            HttpRestResource::WrapperJson(jsonStr, g_exceptionInfo.at(httplib::StatusCode::UnprocessableContent_422)));
        return false;
    }
    auto opt = MakeHttpHeadersOpt(reqCtx->Req());
    if (inferType == MsgType::MSG_TYPE_TGI) {
        DispatchInfer(
            reqCtx, opt, MSG_TYPE_TGI, [opt](std::shared_ptr<SingleLLMReqHandlerBase> singleLLMReqHandlerBase) {
                return std::make_shared<SingleReqTgiTextInferInterface>(singleLLMReqHandlerBase, opt->isReCompute);
            });
    } else {
        DispatchInfer(
            reqCtx, opt, MSG_TYPE_VLLM, [opt](std::shared_ptr<SingleLLMReqHandlerBase> singleLLMReqHandlerBase) {
                return std::make_shared<SingleReqVllmInferInterface>(singleLLMReqHandlerBase, opt->isReCompute);
            });
    }
    return true;
}

void HttpHandler::HandleGeneralTGIPostGenerate(const httplib::Request &request, ReqCtxPtr &reqCtx)
{
    bool streamMode = false;
    std::string error{};
    if (JsonParse::DecodeGeneralTGIStreamMode(reqCtx->MsgBody(), streamMode, error) != EP_OK) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            ABNORMAL_TRANSMISSION_ERROR), "Parse stream failed. json format or value of stream is valid.");
        HttpRestResource::ResponseJsonBody(reqCtx, httplib::StatusCode::UnprocessableContent_422,
            HttpRestResource::WrapperJson(error, g_exceptionInfo.at(httplib::StatusCode::UnprocessableContent_422)));
        return;
    }

    auto opt = MakeHttpHeadersOpt(request);
    if (opt == nullptr) {
        return;
    }
    DispatchInfer(reqCtx, opt, MSG_TYPE_GENERAL_TGI,
                  [opt, streamMode](std::shared_ptr<SingleLLMReqHandlerBase> singleLLMReqHandlerBase) {
                      return std::make_shared<SingleReqGeneralTgiTextInferInterface>(singleLLMReqHandlerBase,
                                                                                     opt->isReCompute, streamMode);
                });
}

int HttpHandler::ManagementInitialize(HttpsServerHelper &server)
{
    // 服务状态
    InitializeServiceStatusResource(server);
    RegisterHealthCheckerHttpHandler(server);
    InitializeServiceMgmtV1(server);
    InitializeServiceMgmtV2(server);
    if (GetServerConfig().inferMode == "dmi") {
        DmiRole::GetInstance()->RunThread();
    }
    return 0;
}

int HttpHandler::BusinessInitialize(HttpsServerHelper &server)
{
    startTime = time(nullptr);
    // register management APIs
    InitializeManagerResource(server);
    // register inference APIs
    InitializeInferResource(server);
    // refer to "EndPoint RESTful interfaces" chapter in Ascend docs for detailed information
    return 0;
}

void HttpHandler::InitializeServiceStatusResource(HttpsServerHelper &server)
{
    /// Self-developed health probe interface.
    ///
    /// This interface is used to check the health status of the service by sending a probe
    /// which simulates real request.
    /// The optional query parameter represented by ((-[0-9]+)?) allows specifying a positive wait time in seconds.
    /// If probe response normally within the wait time, it will return 200;
    /// otherwise, it returns other http status code.
    server.Get(R"(/health/timed((-[0-9]+)?))",
        [](const httplib::Request &request, httplib::Response &response) {
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
        if (StopServiceOption::stopServiceFlag.load()) {
            HttpRestResource::ResponseJsonBody(context, httplib::StatusCode::OK_200,
                HttpRestResource::WrapperStatusJson("healthy"));
            return;
        }
        if (!HealthManager::GetHealth()) {
            HttpRestResource::ResponseNobody(context, httplib::StatusCode::ServiceUnavailable_503);
            return;
        }
        auto g_waitTime = GetUriParameters(context->Req(), 1);
        uint32_t waitTime = 0;
        if (!CheckWaitTime(g_waitTime, waitTime, context)) {
            return;
        }

        // 直接调用 HealthChecker 执行虚推
        HealthChecker &checker = HealthChecker::GetInstance();
        SimulateResult result = checker.RunHttpTimedHealthCheck(waitTime);

        // 将 SimulateResult 转换为 HTTP 响应
        switch (result.status) {
            case SimulateResult::Status::SUCCESS:
            case SimulateResult::Status::BUSY:
                HttpRestResource::ResponseJsonBody(context, httplib::StatusCode::OK_200,
                    HttpRestResource::WrapperStatusJson("healthy"));
                break;
            case SimulateResult::Status::TIMEOUT:
            case SimulateResult::Status::ERROR:
            default:
                HttpRestResource::ResponseJsonBody(context, httplib::StatusCode::InternalServerError_500,
                    HttpRestResource::WrapperJson(result.message,
                        g_exceptionInfo.at(httplib::StatusCode::InternalServerError_500)));
                break;
        }
        return;
    });

    /// Triton health live interface.
    ///
    /// This interface is used to check the health status of the service by health atomic flag and node status.
    /// If service is health, it will return 200;
    /// otherwise, it returns other http status code.
    server.Get("/v2/health/live", [](const httplib::Request &request, httplib::Response &response) {
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
        if (CheckHealthAndStop(context)) {
            return;
        }
        HandleGetLivenessAndReadiness(context);
        return;
    });

    /// Triton health ready interface.
    ///
    /// This interface is used to check the health status of the service by health atomic flag and node status.
    /// If service is health, it will return 200;
    /// otherwise, it returns other http status code.
    server.Get("/v2/health/ready", [](const httplib::Request &request, httplib::Response &response) {
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
        if (CheckHealthAndStop(context)) {
            return;
        }
        HandleGetLivenessAndReadiness(context);
        return;
    });

    /// Triton model health ready interface.
    ///
    /// This interface is used to check the health status of the service by health atomic flag and node status.
    /// The required query parameter represented by MODELS_NAME_RE allows specifying model name,
    /// which must be identical to the model name specified in configuration json file.
    /// If service is health, it will return 200;
    /// otherwise, it returns other http status code.
    server.Get(R"(/v2/models/)" + MODELS_NAME_RE + R"(/ready)",
               [](const httplib::Request &request, httplib::Response &response) {
        ModelDeployConfig config;
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
        if (CheckHealthAndStop(context)) {
            return;
        }
        if (!GetRequestModelConfig(context, config)) {
            return;
        }
        HandleGetHealthStatus(context);
        return;
    });

    /// tgi and vLLM health interface.
    ///
    /// This interface is used to check the health status of the service by health atomic flag and node status.
    /// If service is health, it will return 200;
    /// otherwise, it returns other http status code.
    server.Get("/health", [](const httplib::Request &request, httplib::Response &response) {
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
        if (CheckHealthAndStop(context)) {
            return;
        }
        HandleGetHealthStatus(context);
        return;
    });

    /// tgi query interface.
    ///
    /// This interface is used to query the model and schedule info.
    /// Must be used under healthy status.
    /// Return 200 if success, otherwise return other http status code.
    server.Get("/info", [](const httplib::Request &request, httplib::Response &response) {
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
        if (CheckHealthAndStop(context)) {
            return;
        }
        HandleGetInfoQuery(context);
        return;
    });

    /// Self-developed slots-query interface.
    ///
    /// This interface is used to query the slots logical info in service.
    /// The required query parameter represented by MODELS_NAME_RE allows specifying model name,
    /// which must be identical to the model name specified in configuration json file.
    /// Must be used under healthy status.
    /// Return 200 if success, otherwise return other http status code.
    server.Get(R"(/v2/models/)" + MODELS_NAME_RE + R"(/getSlotCount)",
        [](const httplib::Request &request, httplib::Response &response) {
            std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
        if (CheckHealthAndStop(context)) {
            return;
        }
        HandleGetSlotCount(context);
        return;
    });

    /// Self-developed statistic query interface.
    ///
    /// This interface is used to query the statistic info. e.g. processingInferRequestNum.
    /// Must be used under healthy status.
    /// Return 200 if success, otherwise return other http status code.
    server.Get("/metrics-json", [](const httplib::Request &request, httplib::Response &response) {
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
        if (CheckHealthAndStop(context)) {
            return;
        }
        HandleHttpMetrics(context);
        return;
    });

    /// Self-developed service stop interface.
    ///
    /// This interface is used to stop the service.
    /// Return 200 if success, otherwise return other http status code.
    server.Get("/stopService", [](const httplib::Request &request, httplib::Response &response) {
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
        if (request.has_param("mode") && request.get_param_value("mode") == "Force") {
            ForceStop(context);
        } else {
            StopService(context);
        }
    });
}

void HttpHandler::RegisterHealthCheckerHttpHandler(HttpsServerHelper &server)
{
    server.Get("/v1/engine-server/running-status", [](const httplib::Request &request, httplib::Response &response) {
        ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "GET /v1/engine-server/running-status request received");
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
        HandleGetEngineStatus(context);
        return;
    });

    server.Post("/v1/engine-server/fault-handling-command",
                [](const httplib::Request &request, httplib::Response &response) {
        ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "POST /v1/engine-server/running-status request received");
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
        HandlePostCmdToEngine(context);
        return;
    });
}

void HttpHandler::HandlePostCmdToEngine(const ReqCtxPtr &reqCtx)
{
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Received post command to engine request");
    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Request body: " << reqCtx->MsgBody());

    FaultRecoveryCmd cmdType;
    std::string command;
    if (JsonParse::DecodeFaultRecoveryCmd(reqCtx->MsgBody(), cmdType, command) != EP_OK) {
        std::string jsonStr(R"delimiter({"Error": "cmd_type must be necessary and data type must be integer.",
            "error_type": "validation"})delimiter");
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, CHECK_ERROR),
                   "Validation failed");
        HttpRestResource::ResponseJsonBody(
            reqCtx, httplib::StatusCode::UnprocessableContent_422,
            HttpRestResource::WrapperJson(jsonStr, g_exceptionInfo.at(httplib::StatusCode::UnprocessableContent_422)));
        return;
    }
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Command type decoded: " << command);

    ServiceStatus serviceStatus = HealthChecker::GetInstance().GetServiceStatus();
    RecoverCommandInfo info(command);
    Status status;

    // 匹配cmd和状态简化分支逻辑
    auto callHandler = [cmdType, serviceStatus, &info, &status](
        FaultRecoveryCmd cmd, ServiceStatus statusToCheck,
        void (*handler)(RecoverCommandInfo&, Status&)
    ) -> bool {
        if (cmdType == cmd && serviceStatus == statusToCheck) {
            handler(info, status);
            return true;
        }
        return false;
    };

    bool handled = callHandler(FaultRecoveryCmd::CMD_PAUSE_ENGINE, SERVICE_NORMAL, ExecuteFaultRecoveryPauseCmd)
        || callHandler(FaultRecoveryCmd::CMD_PAUSE_ENGINE, SERVICE_ABNORMAL, ExecuteFaultRecoveryPauseCmd)
        || callHandler(FaultRecoveryCmd::CMD_PAUSE_ENGINE, SERVICE_BUSY, ExecuteFaultRecoveryPauseCmd)
        || callHandler(FaultRecoveryCmd::CMD_PAUSE_ENGINE_ROCE, SERVICE_NORMAL, ExecuteFaultRecoveryPauseCmd)
        || callHandler(FaultRecoveryCmd::CMD_PAUSE_ENGINE_ROCE, SERVICE_ABNORMAL, ExecuteFaultRecoveryPauseCmd)
        || callHandler(FaultRecoveryCmd::CMD_PAUSE_ENGINE_ROCE, SERVICE_BUSY, ExecuteFaultRecoveryPauseCmd)
        || callHandler(FaultRecoveryCmd::CMD_REINIT_NPU, SERVICE_PAUSE, ExecuteFaultRecoveryReinitNpuCmd)
        || callHandler(FaultRecoveryCmd::CMD_START_ENGINE, SERVICE_READY, ExecuteFaultRecoveryStartEngineCmd)
        || callHandler(FaultRecoveryCmd::CMD_START_ENGINE, SERVICE_PAUSE, ExecuteFaultRecoveryStartEngineCmd);
    if (!handled) {
        ULOG_WARN(SUBMODLE_NAME_ENDPOINT,
                  GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_FAULT_CONTROL, CHECK_WARNING),
                  "Command is not consistent with current status.");
        HttpRestResource::ResponseJsonBody(
            reqCtx, httplib::StatusCode::BadRequest_400,
            HttpRestResource::WrapperJson("Command is not consistent with current status.",
                                          g_exceptionInfo.at(httplib::StatusCode::BadRequest_400)));
        return;
    }

    std::string jsonStr;
    JsonParse::EncodeCmdResult(status, info, jsonStr);
    HttpRestResource::ResponseWithBody(reqCtx, httplib::StatusCode::OK_200, "application/json", jsonStr);
}

void HttpHandler::HandleGetEngineStatus(const ReqCtxPtr &requestContext)
{
    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Handling get engine status request");
    ServiceStatus status;
    std::vector<ErrorItem> errorList;
    HealthChecker::GetInstance().GetStatusAndErrorList(status, errorList);
    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Handling get engine status request");
    std::string jsonStr;
    JsonParse::EncodeHealthStatus(status, errorList, jsonStr);
    HttpRestResource::ResponseWithBody(requestContext, httplib::StatusCode::OK_200, "application/json", jsonStr);
}

void HttpHandler::HandleGetLivenessAndReadiness(const ReqCtxPtr &requestContext)
{
    // 检查llmInferEngine是否启动以及从节点是否存在故障
    std::map<std::string, NodeHealthStatus> slaveStatus{};
    Status nodeStatus = GetInferInstance()->GetNodeStatus(slaveStatus);
    if (!nodeStatus.IsOk()) {
        RespondUnhealthy(requestContext,
            "Failed to get node status: " + nodeStatus.StatusMsg());
        return;
    }
    for (auto const& slave : slaveStatus) {
        if (slave.second == NodeHealthStatus::ABNORMAL) {
            std::string jsonStr;
            JsonParse::EncodeAbnormalNodeInfo(slaveStatus, jsonStr);
            HttpRestResource::ResponseJsonBody(requestContext,
                httplib::StatusCode::InternalServerError_500, jsonStr);
            return;
        }
    }

    // PD分离模式下检查连接情况
    if (IsDMI() && !DmiRole::GetInstance()->IsHealthy()) {
        RespondUnhealthy(requestContext, "DMI connection is unhealthy");
        return;
    }

    // 获取基础健康状态
    ServiceStatus serviceStatus = HealthChecker::GetInstance().GetServiceStatus();
    bool isLiveness = requestContext->Path().find("ready") == std::string::npos;

    // 处理响应结果
    HandleHealthResponse(requestContext, serviceStatus, isLiveness);
}

void HttpHandler::HandleHealthResponse(const ReqCtxPtr &ctx, mindie_llm::ServiceStatus status, bool isLiveness)
{
    switch (status) {
        case SERVICE_NORMAL:
            // 正常状态
            HttpRestResource::ResponseNobody(ctx, httplib::StatusCode::OK_200);
            break;
        case SERVICE_BUSY:
            // 繁忙状态：liveness正常；readiness返回503，表示服务暂时不可用
            if (isLiveness) {
                ordered_json jsonObj;
                jsonObj["message"] = "Service is alive but busy";
                HttpRestResource::ResponseJsonBody(ctx,
                    httplib::StatusCode::OK_200, jsonObj.dump());
            } else {
                ordered_json jsonObj;
                jsonObj["message"] = "Service is alive but busy. Consider reducing request frequency";
                HttpRestResource::ResponseJsonBody(ctx,
                    httplib::StatusCode::ServiceUnavailable_503, jsonObj.dump());
            }
            break;
        default:
            // 其余状态服务均不可用
            RespondUnhealthy(ctx, "Service is abnormal");
            break;
    }
}

void HttpHandler::RespondUnhealthy(const ReqCtxPtr &ctx, const std::string& message)
{
    ordered_json jsonObj;
    jsonObj["message"] = message;
    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
        GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER, ABNORMAL_TRANSMISSION_ERROR),
        message);

    // 不健康状态返回500
    HttpRestResource::ResponseJsonBody(ctx, httplib::StatusCode::InternalServerError_500,
        jsonObj.dump());
}

void HttpHandler::ExecuteFaultRecoveryPauseCmd(RecoverCommandInfo &info, Status &status)
{
    // Modify status immediately when receiving Pause command: Normal -> Pause
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Update Status By Code");
    HealthChecker::GetInstance().UpdateStatus(SERVICE_PAUSE);
    status = GetInferInstance()->ControlInferInstance(info);
    if (!status.IsOk()) {
        ULOG_WARN(SUBMODLE_NAME_ENDPOINT,
                  GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_FAULT_CONTROL, RESPONSE_PROCESS_ERROR),
                  "Failed to execute command. " << status.StatusMsg());
        // Pause failed, modify status to abnormal
        ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Update Status By Code");
        HealthChecker::GetInstance().UpdateStatus(SERVICE_ABNORMAL);
    }
}

void HttpHandler::ExecuteFaultRecoveryReinitNpuCmd(RecoverCommandInfo &info, Status &status)
{
    status = GetInferInstance()->ControlInferInstance(info);
    if (!status.IsOk()) {
        // Reinit failed, Pause -> Abnormal
        ULOG_WARN(SUBMODLE_NAME_ENDPOINT,
                  GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_FAULT_CONTROL, RESPONSE_PROCESS_ERROR),
                  "Failed to execute command. " << status.StatusMsg());
        ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Update Status By Code");
        HealthChecker::GetInstance().UpdateStatus(SERVICE_ABNORMAL);
    } else {
        // Reinit -> Ready
        HealthChecker::GetInstance().UpdateStatus(SERVICE_READY);
        ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Update Status By Code");
    }
}

void HttpHandler::ExecuteFaultRecoveryStartEngineCmd(RecoverCommandInfo &info, Status &status)
{
    status = GetInferInstance()->ControlInferInstance(info);
    if (!status.IsOk()) {
        ULOG_WARN(SUBMODLE_NAME_ENDPOINT,
                  GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_FAULT_CONTROL, RESPONSE_PROCESS_ERROR),
                  "Failed to execute command. " << status.StatusMsg());
        HealthChecker::GetInstance().UpdateStatus(SERVICE_ABNORMAL);
    } else {
        // Ready -> Normal
        HealthChecker::GetInstance().UpdateStatus(SERVICE_NORMAL);
    }
}

void HttpHandler::ForceStop(const ReqCtxPtr &requestContext)
{
    if (!StopServiceOption::stopServiceFlag.load()) {
        StopServiceOption::stopServiceFlag.store(true);
    } else {
        ULOG_WARN(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_SPLITWISE,
            CHECK_WARNING), "Service has been stop!");
        HttpRestResource::ResponseNobody(requestContext, httplib::StatusCode::InternalServerError_500);
        return;
    }
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Force stop instance after 5 seconds...");
    std::thread([]() {
        std::this_thread::sleep_for(std::chrono::seconds(WAIT_TIME));
        ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Now terminate instance!");
        killpg(getpgrp(), SIGKILL);
    }).detach();
}

bool HttpHandler::JudgeRestProcess()
{
    std::map<std::string, uint64_t> batchSchedulerMetrics{};
    Status status = GetInferInstance()->GetBatchSchedulerMetrics(batchSchedulerMetrics);
    if (!status.IsOk()) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR,
            SUBMODLE_FEATURE_SPLITWISE,
            LOCAL_INVOKING_ERROR), "Failed to get batchScheduler metrics. "
            << status.StatusMsg());
        return false;
    }
    if (batchSchedulerMetrics.find("processingInferRequestNum") == batchSchedulerMetrics.end() ||
        batchSchedulerMetrics.find("waitingInferRequestNum") == batchSchedulerMetrics.end()) {
        return false;
    }
    if (batchSchedulerMetrics["processingInferRequestNum"] > 0 || batchSchedulerMetrics["waitingInferRequestNum"] > 0) {
        return true;
    }
    return false;
}

void HttpHandler::StopService(const ReqCtxPtr &requestContext)
{
    if (!StopServiceOption::stopServiceFlag.load()) {
        StopServiceOption::stopServiceFlag.store(true);
    } else {
        ULOG_WARN(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_SPLITWISE,
            CHECK_WARNING), "Service has been stop!");
        HttpRestResource::ResponseNobody(requestContext, httplib::StatusCode::InternalServerError_500);
        return;
    }

    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Waiting for running requests to complete...");
    while (HttpHandler::JudgeRestProcess()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(JUDGE_PROCESS_WAIT_TIME));
    }
    HttpRestResource::ResponseNobody(requestContext, httplib::StatusCode::OK_200);
}

void HttpHandler::HandleHttpMetrics(const ReqCtxPtr &requestContext)
{
    std::string jsonStr = "";
    std::map<std::string, uint64_t> batchSchedulerMetrics{};
    Status statusGetMetrics = GetInferInstance()->GetBatchSchedulerMetrics(batchSchedulerMetrics);
    if (!statusGetMetrics.IsOk()) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            ABNORMAL_TRANSMISSION_ERROR), "Failed to get batch scheduler metrics");
        HttpRestResource::ResponseNobody(requestContext, httplib::StatusCode::InternalServerError_500);
        return;
    }
    if (!JsonParse::JsonHttpMetrics(HttpMetrics::GetInstance(), batchSchedulerMetrics, jsonStr)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            LOCAL_INVOKING_ERROR), "Failed to get http metrics");
        HttpRestResource::ResponseNobody(requestContext, httplib::StatusCode::InternalServerError_500);
        return;
    }
    HttpRestResource::ResponseJsonBody(requestContext, httplib::StatusCode::OK_200, jsonStr);
}

void HttpHandler::GetPrometheusMetrics(const ReqCtxPtr &requestContext)
{
    if (!PrometheusMetrics::GetInstance()->IsActivate()) {
        HttpRestResource::ResponseJsonBody(requestContext, httplib::StatusCode::InternalServerError_500,
            HttpRestResource::WrapperJson("Environment variable MIES_SERVICE_MONITOR_MODE is not set.",
            g_exceptionInfo.at(httplib::StatusCode::InternalServerError_500)));
        return;
    }
    std::string prometheusMetricsRes = "";
    PrometheusMetrics::GetInstance()->GetMetricsResult(prometheusMetricsRes);
    HttpRestResource::ResponseWithBody(requestContext, httplib::StatusCode::OK_200, "text/plain; charset=utf-8",
        prometheusMetricsRes);
}

bool CanDmiRoleReqProcess()
{
    const std::string inferMode = GetServerConfig().inferMode;
    if (inferMode == "standard") {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            CHECK_ERROR), "Dmi role request is forbidden in standard mode.");
        return false;
    }
    // Check if need update
    if (GetInferInstance()->GetPDRoleStatus() == PDRoleStatus::SWITCHING) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            CHECK_ERROR), "In dmi mode, this node is initializing.");
        return false;
    }
    return true;
}

void HttpHandler::InitializeServiceMgmtV1(HttpsServerHelper &server)
{
    /// Self-developed config query interface.
    ///
    /// Return the configuration of the current server.
    server.Get(R"(/v1/config)", [](const httplib::Request &request, httplib::Response &response) {
        std::shared_ptr<RequestContext> ctx = std::make_shared<RequestContext>(request, response);
        const ScheduleConfig &scheduleParam = GetScheduleConfig();
        const std::vector<ModelDeployConfig> &modelParam = GetModelDeployConfig();
        OrderedJson jsonObj;
        if (modelParam.empty()) {
            std::string errorResponseStr = "Failed to get modelDeployConfig, at least 1 modelDeployConfig expected.";
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, CHECK_ERROR),
                       errorResponseStr);
            HttpRestResource::ResponseJsonBody(
                ctx, httplib::StatusCode::InternalServerError_500,
                HttpRestResource::WrapperJson(errorResponseStr,
                                              g_exceptionInfo.at(httplib::StatusCode::InternalServerError_500)));
            return;
        }
        jsonObj["modelName"] = modelParam[0].modelName;
        jsonObj["maxSeqLen"] = modelParam[0].maxSeqLen;
        jsonObj["npuMemSize"] = modelParam[0].npuMemSize;
        jsonObj["cpuMemSize"] = modelParam[0].cpuMemSize;
        jsonObj["worldSize"] = modelParam[0].worldSize;
        jsonObj["maxOutputLen"] = scheduleParam.maxIterTimes;
        jsonObj["cacheBlockSize"] = scheduleParam.cacheBlockSize;

        HttpRestResource::ResponseJsonBody(ctx, httplib::StatusCode::OK_200, jsonObj.dump());
        return;
    });

    /// Self-developed status-query interface.
    ///
    /// Get the current status of the service.
    server.Get("/v1/status", [](const httplib::Request &request, httplib::Response &response) {
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
        if (StopServiceOption::stopServiceFlag.load()) {
            HttpRestResource::ResponseJsonBody(context, httplib::StatusCode::ServiceUnavailable_503,
                HttpRestResource::WrapperJson("The service has been stopped.",
                g_exceptionInfo.at(httplib::StatusCode::ServiceUnavailable_503)));
            return;
        }
        if (!HealthManager::GetHealth()) {
            HttpRestResource::ResponseNobody(context, httplib::StatusCode::ServiceUnavailable_503);
            return;
        }
        HandleStatusV1(context);
        return;
    });

    /// Self-developed role-query interface in PD wise mode.
    ///
    /// Get the role(prefill/decode) of current service.
    server.Post(R"(/v1/role/(prefill|decode|flex))", [](const httplib::Request &request, httplib::Response &response) {
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
        std::lock_guard<std::mutex> lock(dmiRoleMutex_);
        if (!CanDmiRoleReqProcess()) {
            OrderedJson jsonObj;
            jsonObj["result"] = "Can't assign DMI role because status is switching.";
            HttpRestResource::ResponseJsonBody(context, httplib::StatusCode::OK_200, jsonObj.dump());
            return;
        }
        std::string roleName = GetUriParameters(context->Req(), 1);
        DmiRole::GetInstance()->HandlePDRoleV1(context, roleName);
        HandlePDWiseUpdateNpuDeviceIds(context);
        return;
    });

    server.Post(R"(/v1/load_lora_adapter)", [](const httplib::Request &request, httplib::Response &response) {
        std::string jsonData;
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
        const std::vector<ModelDeployConfig> &modelConfig = GetModelDeployConfig();
        const std::vector<LoraConfig> &loraConfig = GetLoraConfig();
        OrderedJson jsonObj;

        JsonParse::EncodeOpenAiModels(modelConfig, loraConfig, startTime, jsonData);
        if (!JsonParse::GetContextJsonBody(context, jsonObj)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
                JSON_PARSE_ERROR), "Req body converts to json fail.");
            HttpRestResource::ResponseJsonBody(context, httplib::StatusCode::OK_200, "Req body converts to json fail.");
            return;
        }
        struct LoraConfig {
            std::string loraName;
            std::string loraPath;
            std::string masterModel;
        };
        std::string homePath;
        std::string errMsg;
        mindie_llm::GetHomePath(homePath);
        std::string loraName = jsonObj.value("lora_name", "");
        std::string loraPath = jsonObj.value("lora_path", "");
        std::string masterModel = jsonObj.value("master_model", "");
        std::regex pattern("^[a-zA-Z0-9_-]{1,256}$");
        if (!std::regex_match(loraName, pattern)) {
            HttpRestResource::ResponseJsonBody(context, httplib::StatusCode::ServiceUnavailable_503,
                                               "Input invalid lora name");
            return;
        }
        std::regex path_pattern(R"(^(\/(?:[\w\-\.]+\/)*[\w\-\.]*\/?)?$|^(?:[\w\-\.]+\/)*[\w\-\.]*\/?$)");
        if (!std::regex_match(loraPath, path_pattern) || loraPath.find("..") != std::string::npos) {
            HttpRestResource::ResponseJsonBody(context, httplib::StatusCode::ServiceUnavailable_503,
                                               "Input invalid lora path");
            return;
        }
        std::vector<LoraParamSPtr> lora_params = {std::make_shared<LoraParam>(loraName, loraPath, masterModel)};
        Status status = GetInferInstance()->HandleLora(LoraOperation::LORA_LOAD, lora_params);
        HttpRestResource::ResponseJsonBody(context, httplib::StatusCode::OK_200, status.StatusMsg());
        return;
    });

    server.Post(R"(/v1/unload_lora_adapter)", [](const httplib::Request &request, httplib::Response &response) {
        std::string jsonData;
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
        auto &modelConfig = GetModelDeployConfig();
        auto &loraConfig = GetLoraConfig();
        OrderedJson jsonObj;
        JsonParse::EncodeOpenAiModels(modelConfig, loraConfig, startTime, jsonData);
        if (!JsonParse::GetContextJsonBody(context, jsonObj)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
                JSON_PARSE_ERROR), "Req body converts to json fail.");
            HttpRestResource::ResponseJsonBody(context, httplib::StatusCode::OK_200, "Req body converts to json fail.");
            return;
        }
        struct LoraConfig {
            std::string loraName;
            std::string loraPath;
            std::string masterModel;
        };
        std::string loraName = jsonObj.value("lora_name", "");
        std::string loraPath = "";
        std::string masterModel = "";
        std::regex pattern("^[a-zA-Z0-9_-]{1,256}$");
        if (!std::regex_match(loraName, pattern)) {
            HttpRestResource::ResponseJsonBody(context, httplib::StatusCode::ServiceUnavailable_503,
                                               "Input invalid lora name");
            return;
        }
        std::vector<LoraParamSPtr> lora_params = {std::make_shared<LoraParam>(loraName, loraPath, masterModel)};
        Status status = GetInferInstance()->HandleLora(LoraOperation::LORA_UNLOAD, lora_params);
        HttpRestResource::ResponseJsonBody(context, httplib::StatusCode::OK_200, status.StatusMsg());
        return;
    });
}

void HttpHandler::InitializeServiceMgmtV2(HttpsServerHelper &server)
{
    /// Self-developed status-query interface.
    ///
    /// Get the current status of the service.
    server.Get("/v2/status", [](const httplib::Request &request, httplib::Response &response) {
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
        if (StopServiceOption::stopServiceFlag.load()) {
            HttpRestResource::ResponseJsonBody(context, httplib::StatusCode::ServiceUnavailable_503,
                HttpRestResource::WrapperJson("The service has been stopped.",
                g_exceptionInfo.at(httplib::StatusCode::ServiceUnavailable_503)));
            return;
        }
        if (!HealthManager::GetHealth()) {
            HttpRestResource::ResponseNobody(context, httplib::StatusCode::ServiceUnavailable_503);
            return;
        }
        HandleStatusV2(context);
        return;
    });

    /// Self-developed role-query interface in PD wise mode.
    ///
    /// Get the role(prefill/decode) of current service.
    server.Post(R"(/v2/role/(prefill|decode))", [](const httplib::Request &request, httplib::Response &response) {
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
        std::lock_guard<std::mutex> lock(dmiRoleMutex_);
        if (!CanDmiRoleReqProcess()) {
            OrderedJson jsonObj;
            jsonObj["result"] = "Can't assign DMI role because status is switching.";
            HttpRestResource::ResponseJsonBody(context, httplib::StatusCode::OK_200, jsonObj.dump());
            return;
        }
        std::string roleName = GetUriParameters(context->Req(), 1);
        DmiRole::GetInstance()->HandlePDRoleV2(context, roleName);
        HandleUpdateNpuDeviceIds(context);
        return;
    });
}

void HttpHandler::HandlePDWiseUpdateNpuDeviceIds(const ReqCtxPtr &ctx)
{
    try {
        ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Handle update npu device ids request for V1 format");
        ordered_json body;
        std::string msgBody = ctx->MsgBody();
        if (!ordered_json::accept(msgBody)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                JSON_PARSE_ERROR), "Convert string to json object failed, CallbackId is " << ctx->CallbackId());
            return;
        }
        body = ordered_json::parse(msgBody);
        std::set<int> npuDeviceIds;
        // 处理V1格式：直接从local["device"]中提取
        if (body.contains("local") && body["local"].contains("device")) {
            for (auto& item : body["local"]["device"]) {
                npuDeviceIds.insert(std::stoi(item["device_logical_id"].get<std::string>()));
            }
        } else {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                JSON_PARSE_ERROR),
                "V1 format error: missing local or device field, CallbackId is " << ctx->CallbackId());
            return;
        }
        
        ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "V1 npu device ids size: " << npuDeviceIds.size());
        HealthChecker::GetInstance().UpdateNpuDeviceIds(npuDeviceIds);
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            JSON_PARSE_ERROR), "Convert string to json object exception, CallbackId is " << ctx->CallbackId());
        return;
    }
}

void HttpHandler::HandleUpdateNpuDeviceIds(const ReqCtxPtr &ctx)
{
    try {
        ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Handle update npu device ids request");
        ordered_json body;
        std::string msgBody = ctx->MsgBody();
        if (!ordered_json::accept(msgBody)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                JSON_PARSE_ERROR), "Convert string to json object failed, CallbackId is " << ctx->CallbackId());
            return;
        }
        body = ordered_json::parse(msgBody, CheckOrderedJsonDepthCallback);
        std::set<int> npuDeviceIds;
        for (auto item: body["local"][0]["dp_inst_list"][0]["device"]) {
            npuDeviceIds.insert(std::stoi(item["device_logical_id"].get<std::string>()));
        }
        ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "npu device ids size: " << npuDeviceIds.size());
        HealthChecker::GetInstance().UpdateNpuDeviceIds(npuDeviceIds);
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            JSON_PARSE_ERROR), "Convert string to json object exception, CallbackId is " << ctx->CallbackId());
        return;
    }
}

int HttpHandler::HandleGetSlotCount(const ReqCtxPtr &ctx)
{
    ModelDeployConfig modelConfig;
    if (GetRequestModelConfig(ctx, modelConfig)) {
        std::string jsonData;
        // total_slots
        const ScheduleConfig &scheduleParam = GetScheduleConfig();
        // free_slots
        uint64_t freeSlot = 0;
        if (!GetAvalSlotNum(freeSlot).IsOk()) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                ABNORMAL_TRANSMISSION_ERROR), "Failed to get free slots");
            HttpRestResource::ResponseNobody(ctx, httplib::StatusCode::InternalServerError_500);
            return httplib::StatusCode::InternalServerError_500;
        }
        // available_tokens_length
        uint64_t remainBlocks = 0;
        if (!GetRemainBlockNum(remainBlocks).IsOk()) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                ABNORMAL_TRANSMISSION_ERROR), "Failed to get request block quotas");
            HttpRestResource::ResponseNobody(ctx, httplib::StatusCode::InternalServerError_500);
            return httplib::StatusCode::InternalServerError_500;
        }
        const ScheduleConfig &scheduleConfig = GetScheduleConfig();
        uint64_t availableTokensLen = remainBlocks * scheduleConfig.cacheBlockSize;
        // generate json
        JsonParse::EncodeSlotCount(scheduleParam, freeSlot, availableTokensLen, jsonData);
        HttpRestResource::ResponseJsonBody(ctx, httplib::StatusCode::OK_200, jsonData);
    }
    return 0;
}

void HttpHandler::HandleTokenizer(const ReqCtxPtr &ctx)
{
    ordered_json body;
    auto errStat = httplib::StatusCode::UnprocessableContent_422;
    if (!JsonParse::GetContextJsonBody(ctx, body)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
            JSON_PARSE_ERROR), "Req body converts to json fail.");
        HttpRestResource::ResponseJsonBody(ctx, errStat,
            HttpRestResource::WrapperJson("Req body converts to json fail.", g_exceptionInfo.at(errStat)));
        return;
    }
    if (!body.contains("inputs") || body["inputs"].is_null() || !body["inputs"].is_string()) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
            CHECK_ERROR), "Inputs in request body is invalid.");
        HttpRestResource::ResponseJsonBody(ctx, errStat,
            HttpRestResource::WrapperJson("Inputs in request body is invalid.", g_exceptionInfo.at(errStat)));
        return;
    }
    
    bool doDecode = true;
    if (body.contains("do_decode") && body["do_decode"].is_boolean()) {
        doDecode = body["do_decode"].get<bool>();
    }
    const std::string inputText = body["inputs"];
    std::string errorMsg = "";
    std::u16string utf16 = GetU16Str(inputText, &errorMsg);
    if (!errorMsg.empty()) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER, CHECK_ERROR),
            errorMsg);
        HttpRestResource::ResponseJsonBody(ctx, errStat,
            HttpRestResource::WrapperJson("Failed to check the input text." + errorMsg, g_exceptionInfo.at(errStat)));
        return;
    }
    if (utf16.length() == 0 || utf16.length() > GetMaxInputLen()) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER, CHECK_ERROR),
            "Inputs must be necessary and data type must be string and length in[0, " << GetMaxInputLen() << "]");
        HttpRestResource::ResponseJsonBody(ctx, errStat,
            HttpRestResource::WrapperJson("Inputs data length in [0, " + std::to_string(GetMaxInputLen()) +
            "], but the length of inputs is " + std::to_string(utf16.length()), g_exceptionInfo.at(errStat)));
        return;
    }

    int numTokenId = 0;
    std::vector<std::string> tokens = {};

    if (utf16.length() > 0) {
        auto status = TokenizerProcessPool::GetInstance().TikToken(inputText, numTokenId, tokens, doDecode);
        if (!status.IsOk()) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
                ABNORMAL_TRANSMISSION_ERROR), "TikToken process fail.");
            HttpRestResource::ResponseJsonBody(ctx, errStat,
                HttpRestResource::WrapperJson("TikToken process fail.", g_exceptionInfo.at(errStat)));
            return;
        }
    }

    ordered_json jsonObj;
    jsonObj["token_number"] = numTokenId;
    jsonObj["tokens"] = tokens;

    HttpRestResource::ResponseJsonBody(ctx, httplib::StatusCode::OK_200, jsonObj.dump());
}

void HttpHandler::HandleGetHealthStatus(const ReqCtxPtr &ctx)
{
    std::map<std::string, NodeHealthStatus> slaveStatus{};
    Status status = GetInferInstance()->GetNodeStatus(slaveStatus);
    if (!status.IsOk()) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
            ABNORMAL_TRANSMISSION_ERROR), "Failed to get health status. " << status.StatusMsg());
        HttpRestResource::ResponseJsonBody(ctx, httplib::StatusCode::InternalServerError_500,
            HttpRestResource::WrapperJson("Failed to get health status.",
            g_exceptionInfo.at(httplib::StatusCode::InternalServerError_500)));
        return;
    }
    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "GetNodeStatus, slave size is " << slaveStatus.size());
    bool statusFlag = true;
    for (auto const& slave : slaveStatus) {
        if (slave.second == NodeHealthStatus::ABNORMAL) {
            ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "SlaveStatus is not ready");
            statusFlag = false;
            break;
        }
    }

    // check if exsits already linked peers in dmi mode
    if (IsDMI()) {
        statusFlag = statusFlag && DmiRole::GetInstance()->IsHealthy();
    }
    if (statusFlag) {
        ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "SlaveStatus is ready");
        HttpRestResource::ResponseNobody(ctx, httplib::StatusCode::OK_200);
        return;
    }
    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "SlaveStatus is abnormal");
    std::string jsonStr;
    JsonParse::EncodeAbnormalNodeInfo(slaveStatus, jsonStr);
    HttpRestResource::ResponseJsonBody(ctx, httplib::StatusCode::InternalServerError_500,
        HttpRestResource::WrapperJson(jsonStr, g_exceptionInfo.at(httplib::StatusCode::InternalServerError_500)));
}

void HttpHandler::InitializeManagerResource(HttpsServerHelper &server)
{
    // /v1 contains OpenAI interfaces and their extensions
    InitializeManagerResourceV1(server);
    // /v2 contains Triton interface and their extensions
    InitializeManagerResourceV2(server);
}

void HttpHandler::InitializeManagerResourceV1(HttpsServerHelper &server)
{
    /// Models info query interface from OpenAI.
    ///
    /// Return models info such as id, object, created, owned_by
    server.Get("/v1/models", [](const httplib::Request &request, httplib::Response &response) {
        std::string jsonData;
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
        const std::vector<ModelDeployConfig> &modelConfig = GetModelDeployConfig();
        const std::vector<LoraConfig> &loraConfig = GetLoraConfig();
        JsonParse::EncodeOpenAiModels(modelConfig, loraConfig, startTime, jsonData);
        HttpRestResource::ResponseJsonBody(context, -1, jsonData);

        std::lock_guard<std::mutex> lock(dmiRoleMutex_);
        std::string loraName = "";
        std::string loraPath = "";
        std::string masterModel = "";

        std::vector<LoraParamSPtr> lora_params = {std::make_shared<LoraParam>(loraName, loraPath, masterModel)};
        Status stats = GetInferInstance()->HandleLora(LoraOperation::LORA_QUERY, lora_params);
        std::vector<mindie_llm::LoraConfig> loraConfigs;
        loraConfigs.reserve(lora_params.size() + 1);

        mindie_llm::LoraConfig masterConfig;
        masterConfig.loraName = modelConfig[0].modelName;
        masterConfig.loraPath = modelConfig[0].modelWeightPath;
        masterConfig.baseModel = "null";
        loraConfigs.emplace_back(masterConfig);

        for (const auto& param_ptr : lora_params) {
            if (!param_ptr) {  // 空指针检查
                continue;
            }
            mindie_llm::LoraConfig signalConfig;
            signalConfig.loraName = param_ptr->loraName;
            signalConfig.loraPath = param_ptr->loraPath;
            signalConfig.baseModel = param_ptr->masterModel;
            loraConfigs.emplace_back(signalConfig);
        }
        JsonParse::EncodeOpenAiModels(modelConfig, loraConfigs, startTime, jsonData);
        using json = nlohmann::json;
        json jsonData2;
        jsonData2["object"] = "list";
        jsonData2["data"] = json::array();
        for (const auto& singleLoraParam : loraConfigs) {
            json dataItem;
            dataItem["id"] = singleLoraParam.loraName;
            dataItem["object"] = "model";
            dataItem["root"] = FileUtils::GetSafeRelativePath(singleLoraParam.loraPath);
            dataItem["owned_by"] = "MindIE Server";
            if (singleLoraParam.baseModel == "") {
                dataItem["parent"] = GetModelDeployConfig()[0].modelName;
            } else {
                dataItem["parent"] = singleLoraParam.baseModel;
            }
            jsonData2["data"].push_back(dataItem);
        }
        std::string formattedJson = jsonData2.dump(4);
        HttpRestResource::ResponseJsonBody(context, httplib::StatusCode::OK_200, formattedJson);
        return;
    });

    /// Models info query interface from OpenAI with selected model name.
    ///
    /// Return models info such as id, object, created, owned_by
    server.Get(R"(/v1/models/)" + MODELS_NAME_RE, [](const httplib::Request &request, httplib::Response &response) {
        ModelDeployConfig config;
        LoraConfig loraConfig;
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);

        std::string jsonData;
         // Find in lora models first and then in base models. Error is record in the "base model" stage.
        if (GetRequestModelConfig(context, loraConfig)) {
            JsonParse::EncodeOpenAiModel(loraConfig, startTime, jsonData);
            HttpRestResource::ResponseJsonBody(context, -1, jsonData);
        } else if (GetRequestModelConfig(context, config)) {
            JsonParse::EncodeOpenAiModel(config, startTime, jsonData);
            HttpRestResource::ResponseJsonBody(context, -1, jsonData);
        }
        return;
    });
}

void HttpHandler::InitializeManagerResourceV2(HttpsServerHelper &server)
{
    /// Trion interface: query the metadata of the service.
    ///
    /// Return the service name, version and other info.
    server.Get("/v2", [](const httplib::Request &request, httplib::Response &response) {
        std::string jsonData;
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
        const ScheduleConfig &config = GetScheduleConfig();
        JsonParse::EncodeTritonEngine(config, jsonData);
        HttpRestResource::ResponseJsonBody(context, -1, jsonData);
        return;
    });

    /// Triton's query interface that returns data info.
    ///
    /// Return acceptable data type of input and output.
    server.Get(R"(/v2/models/)" + MODELS_NAME_RE, [](const httplib::Request &request, httplib::Response &response) {
        ModelDeployConfig config;
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
        if (GetRequestModelConfig(context, config)) {
            std::string jsonData;
            JsonParse::EncodeTritonModel(config, jsonData);
            HttpRestResource::ResponseJsonBody(context, -1, jsonData);
        }
        return;
    });

    /// Triton's query interface that returns model info.
    ///
    /// Return model info such as max_seq_len and npu_mem_size.
    server.Get(R"(/v2/models/)" + MODELS_NAME_RE + R"(/config)",
        [](const httplib::Request &request, httplib::Response &response) {
            ModelDeployConfig config;
            std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
            if (GetRequestModelConfig(context, config)) {
                std::string jsonData;
                JsonParse::EncodeTritonModelConfig(config, jsonData);
                HttpRestResource::ResponseJsonBody(context, -1, jsonData);
            }
            return;
        });
}

void HttpHandler::InitializeInferResourceV2(HttpsServerHelper &server)
{
    /// Trion's inference interface that takes token input.
    ///
    /// Return inference result, for more information please refer to MINDIE documentation.
    server.Post(R"(/v2/models/)" + MODELS_NAME_RE + R"(/infer)",
        [&server](const httplib::Request &request, httplib::Response &response) {
            ModelDeployConfig config;
            std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);

            server.AddRequestToMonitor(context);
            if (IsDMI() && !CheckDMIReqValid(request, context)) {
                return;
            }
            if (!GetRequestModelConfig(context, config)) {
                return;
            }

            auto opt = MakeHttpHeadersOpt(request);
            DispatchInfer(context, opt, MSG_TYPE_TRITON_TOKEN,
                          [opt](std::shared_ptr<SingleLLMReqHandlerBase> singleLLMReqHandlerBase) {
                              return std::make_shared<SingleReqTritonTokenInferInterface>(singleLLMReqHandlerBase,
                                                                                          opt->isReCompute);
                        });
            return;
        });

    /// Self-developed interface used to stop inference request.
    ///
    /// Stop the inference request if it is not finished yet.
    server.Post(R"(/v2/models/)" + MODELS_NAME_RE + R"(/stopInfer)",
        [&server](const httplib::Request &request, httplib::Response &response) {
            ModelDeployConfig config;
            std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
            if (!GetRequestModelConfig(context, config)) {
                return;
            }
            std::shared_ptr<SingleLLMPnDReqHandler> singleLLMPnDReqHandler =
                std::make_shared<SingleLLMPnDReqHandler>(context);
            auto inferInterface = std::make_shared<SingleReqTritonTextInferInterface>(singleLLMPnDReqHandler, false,
                                                                                      GetUriParameters(request, 1));
            inferInterface->Stop();
            return;
        });

    /// Triton's non-stream inferece interface that accept string as input.
    ///
    /// Return inference result, for more information please refer to MINDIE documentation.
    server.Post(R"(/v2/models/)" + MODELS_NAME_RE + R"(/generate)",
        [&server](const httplib::Request &request, httplib::Response &response) {
            ModelDeployConfig config;
            std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
            server.AddRequestToMonitor(context);
            if (IsDMI() && !CheckDMIReqValid(request, context)) {
                return;
            }
            if (!GetRequestModelConfig(context, config)) {
                return;
            }

            auto opt = MakeHttpHeadersOpt(request);
            DispatchInfer(
                context, opt, MSG_TYPE_TRITON,
                [opt, &request](std::shared_ptr<SingleLLMReqHandlerBase> singleLLMReqHandlerBase) {
                    return std::make_shared<SingleReqTritonTextInferInterface>(singleLLMReqHandlerBase, false,
                                                                GetUriParameters(request, 1), opt->isReCompute);
                });
            return;
        });

    /// Triton's stream inferece interface that accept string as input.
    ///
    /// Return inference result, for more information please refer to MINDIE documentation.
    server.Post(R"(/v2/models/)" + MODELS_NAME_RE + R"(/generate_stream)",
        [&server](const httplib::Request &request, httplib::Response &response) {
            ModelDeployConfig config;
            std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
            server.AddRequestToMonitor(context);
            if (IsDMI() && !CheckDMIReqValid(request, context)) {
                return;
            }
            if (!GetRequestModelConfig(context, config)) {
                return;
            }

            auto opt = MakeHttpHeadersOpt(request);
            DispatchInfer(context, opt, MSG_TYPE_TRITON,
                          [opt, &request](std::shared_ptr<SingleLLMReqHandlerBase> singleLLMReqHandlerBase) {
                              return std::make_shared<SingleReqTritonTextInferInterface>(
                                  singleLLMReqHandlerBase, true, GetUriParameters(request, 1), opt->isReCompute);
                        });
            return;
        });

    /// Self-developed interface to simulate tokenizer.
    ///
    /// Returns count of tokens that transfered from text by tokenizer.
    server.Post(R"(/v1/tokenizer)", [](const httplib::Request &request, httplib::Response &response) {
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
        HandleTokenizer(context);
        return;
    });
}

void HttpHandler::InitializeInferResource(HttpsServerHelper &server)
{
    /// tgi inference interface.
    ///
    /// Return inference result, for more information please refer to MINDIE documentation.
    server.Post("/", [&server](const httplib::Request &request, httplib::Response &response) {
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
        server.AddRequestToMonitor(context);
        if (IsDMI() && !CheckDMIReqValid(request, context)) {
            return;
        }
        HandleGeneralTGIPostGenerate(request, context);
        return;
    });

    /// tgi and vLLM inference interface.
    ///
    /// Return inference result, for more information please refer to MINDIE documentation.
    server.Post("/generate", [&server](const httplib::Request &request, httplib::Response &response) {
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
        server.AddRequestToMonitor(context);
        if (IsDMI() && !CheckDMIReqValid(request, context)) {
            return;
        }
        HandlePostGenerate(context);
        return;
    });

    /// tgi stream inference interface.
    ///
    /// Return inference result, for more information please refer to MINDIE documentation.
    server.Post("/generate_stream", [&server](const httplib::Request &request, httplib::Response &response) {
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
        server.AddRequestToMonitor(context);
        if (IsDMI() && !CheckDMIReqValid(request, context)) {
            return;
        }

        auto opt = MakeHttpHeadersOpt(request);
        DispatchInfer(context, opt, MSG_TYPE_TGI,
                      [opt](std::shared_ptr<SingleLLMReqHandlerBase> singleLLMReqHandlerBase) {
                          return std::make_shared<SingleReqTgiTextInferInterface>(singleLLMReqHandlerBase,
                                                                                  opt->isReCompute, true);
                    });
        return;
    });

    /// OpenAI chat inference interface.
    ///
    /// Return inference result, for more information please refer to MINDIE documentation.
    server.Post("/v1/chat/completions", [&server](const httplib::Request &request, httplib::Response &response) {
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
        server.AddRequestToMonitor(context);
        if (IsDMI() && !CheckDMIReqValid(request, context)) {
            return;
        }

        const ServerConfig &config = GetServerConfig();
        std::shared_ptr<SingleLLMReqHandlerBase> singleLLMReqHandlerBase;
        std::shared_ptr<SingleReqInferInterfaceBase> openAIChatInfer;
        uint16_t msgType = config.openAiSupportedvLLM ? MSG_TYPE_VLLM_OPENAI : MSG_TYPE_OPENAI;
        auto opt = MakeHttpHeadersOpt(request);
        if (opt->reqType == InferReqType::REQ_PREFILL) {
            // PD分离
            singleLLMReqHandlerBase = std::make_shared<SingleLLMPrefillReqHandler>(context, msgType, opt->isReCompute);
        } else {
            // 非PD分离
            singleLLMReqHandlerBase = std::make_shared<SingleLLMPnDReqHandler>(context, opt->isFlexLocal);
        }

        // 查询modelname是否在loralist里
        std::vector<LoraParamSPtr> lora_params = {std::make_shared<LoraParam>("", "", "")};
        Status stats = GetInferInstance()->HandleLora(LoraOperation::LORA_QUERY, lora_params);
        std::vector<mindie_llm::LoraConfig> loraConfigs;
        loraConfigs.reserve(lora_params.size());
        for (const auto& param_ptr : lora_params) {
            if (!param_ptr) {  // 空指针检查
                continue;
            }
            mindie_llm::LoraConfig signalConfig;
            signalConfig.loraName = param_ptr->loraName;
            signalConfig.loraPath = param_ptr->loraPath;
            signalConfig.baseModel = param_ptr->masterModel;
            loraConfigs.emplace_back(signalConfig);
        }

        if (config.openAiSupportedvLLM) {
            ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "OpenAI support by vLLM process.");
            openAIChatInfer = std::make_shared<SingleReqVllmOpenAiInferInterface>(singleLLMReqHandlerBase,
                                                                                  opt->isReCompute, lora_params);
        } else {
            ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "OpenAI support by original process.");
            openAIChatInfer =
                std::make_shared<SingleReqOpenAiInferInterface>(singleLLMReqHandlerBase, opt->isReCompute, lora_params);
        }

        if (opt->reqType == InferReqType::REQ_STAND_INFER) {
            RequestIdNew inferRequestId = openAIChatInfer->GetRequestId();
            context->SetStopInferFunction([inferRequestId]() { // Set Stop Infer Handler
                Status status = GetInferInstance()->ControlRequest(inferRequestId, OperationV2::STOP);
            });
        }
        openAIChatInfer->Process();
    });

    /// OpenAi completions inference interface.
    ///
    /// Return inference result, for more information please refer to MINDIE documentation.
    server.Post("/v1/completions", [&server](const httplib::Request &request, httplib::Response &response) {
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
        server.AddRequestToMonitor(context);
        if (IsDMI() && !CheckDMIReqValid(request, context)) {
            return;
        }

        std::shared_ptr<SingleLLMReqHandlerBase> singleLLMReqHandlerBase;
        std::shared_ptr<SingleReqInferInterfaceBase> openAIChatInfer;
        auto opt = MakeHttpHeadersOpt(request);
        if (opt->reqType == InferReqType::REQ_PREFILL) {
            // PD分离
            singleLLMReqHandlerBase =
                std::make_shared<SingleLLMPrefillReqHandler>(context, MSG_TYPE_VLLM_OPENAI_COMP, opt->isReCompute);
        } else {
            // 非PD分离
            singleLLMReqHandlerBase = std::make_shared<SingleLLMPnDReqHandler>(context, opt->isFlexLocal);
        }

        // 查询modelname是否在loralist里
        std::vector<LoraParamSPtr> lora_params = {std::make_shared<LoraParam>("", "", "")};
        Status stats = GetInferInstance()->HandleLora(LoraOperation::LORA_QUERY, lora_params);
        std::vector<mindie_llm::LoraConfig> loraConfigs;
        loraConfigs.reserve(lora_params.size());
        for (const auto& param_ptr : lora_params) {
            if (!param_ptr) {  // 空指针检查
                continue;
            }
            mindie_llm::LoraConfig signalConfig;
            signalConfig.loraName = param_ptr->loraName;
            signalConfig.loraPath = param_ptr->loraPath;
            signalConfig.baseModel = param_ptr->masterModel;
            loraConfigs.emplace_back(signalConfig);
        }
        ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Open ai support by vLLM process.");
        openAIChatInfer = std::make_shared<SingleReqVllmOpenAiCompletionsInferInterface>(singleLLMReqHandlerBase,
                                                                                         opt->isReCompute, lora_params);
        if (opt->reqType == InferReqType::REQ_STAND_INFER) {
            RequestIdNew inferRequestId = openAIChatInfer->GetRequestId();
            context->SetStopInferFunction([inferRequestId]() {
                Status status = GetInferInstance()->ControlRequest(inferRequestId, OperationV2::STOP);
            });
        }
        openAIChatInfer->Process();
    });

    /// Self-developed inference interface that takes text input.
    ///
    /// Return inference result, for more information please refer to MINDIE documentation.
    server.Post("/infer", [&server](const httplib::Request &request, httplib::Response &response) {
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);

        server.AddRequestToMonitor(context);
        if (IsDMI() && !CheckDMIReqValid(request, context)) {
            return;
        }

        auto opt = MakeHttpHeadersOpt(request);
        DispatchInfer(
            context, opt, MSG_TYPE_INFER, [opt](std::shared_ptr<SingleLLMReqHandlerBase> singleLLMReqHandlerBase) {
                return std::make_shared<SingleReqSelfDevelopInferInterface>(singleLLMReqHandlerBase, opt->isReCompute);
            });
        return;
    });

    /// Self-developed inference interface that takes tokens input.
    ///
    /// Return inference result, for more information please refer to MINDIE documentation.
    server.Post("/infer_token", [&server](const httplib::Request &request, httplib::Response &response) {
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);

        server.AddRequestToMonitor(context);
        if (IsDMI() && !CheckDMIReqValid(request, context)) {
            return;
        }

        auto opt = MakeHttpHeadersOpt(request);
        DispatchInfer(
            context, opt, MSG_TYPE_INFER, [opt](std::shared_ptr<SingleLLMReqHandlerBase> singleLLMReqHandlerBase) {
                return std::make_shared<SingleReqSelfDevelopInferInterface>(singleLLMReqHandlerBase, opt->isReCompute);
            });
        return;
    });

    /// Self-developed inference interface used by PD wise mode.
    ///
    /// Return inference result, for more information please refer to MINDIE documentation.
    server.Get("/dresult", [&server](const httplib::Request &request, httplib::Response &response) {
        ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Create new D result connection");
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
        if (!CheckDresultReqValid(context)) {
            return;
        }
        HandleDResult(context);
        if (!keepAlive.load()) {
            keepAlive.store(true);
            std::thread keepAliveThread(DResultKeepAlive);
            keepAliveThread.detach();
        }
        return;
    });

    InitializeInferResourceV2(server);
}

bool HttpHandler::GetRequestModelConfig(const ReqCtxPtr &requestContext, ModelDeployConfig &config)
{
    auto modelName = GetUriParameters(requestContext->Req(), 1);
    const std::vector<ModelDeployConfig> &modelParams = GetModelDeployConfig();
    for (auto &modelParam : modelParams) {
        if (modelName == modelParam.modelName) {
            config = modelParam;
            return true;
        }
    }

    auto desc = std::string("Model ").append(modelName).append(" not found.");
    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SERVER_REQUEST,
               CHECK_ERROR), desc);
    HttpRestResource::ResponseJsonBody(requestContext, httplib::StatusCode::NotFound_404,
        HttpRestResource::WrapperJson(desc, g_exceptionInfo.at(httplib::StatusCode::NotFound_404)));
    return false;
}


bool HttpHandler::GetRequestModelConfig(const ReqCtxPtr &requestContext, LoraConfig &config)
{
    auto modelName = GetUriParameters(requestContext->Req(), 1);
    const std::vector<LoraConfig> &loraParams = GetLoraConfig();
    for (auto &loraParam : loraParams) {
        if (modelName == loraParam.loraName) {
            config = loraParam;
            return true;
        }
    } // Don't send error msg if only model not found in lora. Will go to find in models and deal with error msg there.

    return false;
}

bool HttpHandler::IsDMI()
{
    return GetServerConfig().inferMode == INFER_MODE_DMI;
}

bool HttpHandler::IsPrefillRole(std::string &reqError)
{
    std::string pdRoleName = GetInferInstance()->GetPDRole();
    if (pdRoleName == "prefill" ||
        (pdRoleName == "flex" && FlexPPercentageProcessor::GetInstance().GetPdRoleFlexPPercentage() != 0)) {
        return true;
    }
    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, CHECK_ERROR),
               "DMI inference requests can be sent only to the prefill node or p_percentage of the flex node == 0");
    reqError = "DMI inference requests can be sent only to the prefill node or p_percentage of the flex node == 0";
    return false;
}

bool HttpHandler::IsAllDMIHeadersExist(const httplib::Request &request, std::string &reqError)
{
    if (request.has_header("req-type") && request.has_header("req-id") && request.has_header("d-target")) {
        return true;
    }
    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
        CHECK_ERROR), "DMI request must have req-type,req-id, d-target headers");
    reqError = "DMI request must have req-type, req-id, d-target headers";
    return false;
}

bool HttpHandler::IsReqIdValid(const httplib::Request &request, std::string &reqError)
{
    std::string reqId = request.get_header_value("req-id");
    // Check if the `reqId` only contains hex digit character
    bool isValidHex = std::all_of(reqId.begin(), reqId.end(), [](char ch) {
        return std::isxdigit(ch);
    });
    if (!isValidHex) {
        reqError = "Invalid req-id, the character of req-id should be hex char [0-9, a-f, A-F], but got " + reqId;
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            CHECK_ERROR), reqError);
        return false;
    }

    size_t maxReqIdLen = 1025;
    if (reqId.length() < maxReqIdLen) {
        return true;
    }
    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
        CHECK_ERROR), "Invalid req-id, the length of req-id cannot exceed 1024");
    reqError = "The length of req-id cannot exceed 1024";
    return false;
}

bool HttpHandler::IsReqTypeValid(const httplib::Request &request, std::string &reqError)
{
    std::string reqType = request.get_header_value("req-type");
    if (reqType == "prefill") {
        return true;
    }
    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
        CHECK_ERROR), "Invalid req-type, req-type must be prefill but got " << reqType);
    reqError = "Parameter req-type must be prefill but got ";
    reqError += reqType;
    return false;
}

bool HttpHandler::IsDTargetValid(const httplib::Request &request, std::string &reqError)
{
    //  dTarget format : "{ip_address} ; {port}"
    std::string dTarget = request.get_header_value("d-target");
    size_t delimiterIndex = dTarget.find(IP_PORT_DELIMITER);
    delimiterIndex = (delimiterIndex == std::string::npos) ? dTarget.size() : delimiterIndex;

    std::string dTargetIp = dTarget.substr(0, delimiterIndex);
    if (!mindie_llm::CheckIp(dTargetIp, "d-target", false)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            CHECK_ERROR), "Invalid d-target, which is " << dTarget);
        reqError = "Parameter d-target must be an IPv4 address";
        return false;
    }

    return true;
}

bool HttpHandler::IsRecomputeParamValid(const httplib::Request &request, std::string &reqError)
{
    std::string recompute = request.get_header_value("is-recompute");
    if (recompute == "" || recompute == "true" || recompute == "false") {
        return true;
    }
    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
        CHECK_ERROR), "Invalid is-recompute param, which is " << recompute);
    reqError = "Parameter is-recompute is an optional parameter. When it is set, it must be true or false.";
    return false;
}

bool HttpHandler::CheckDMIReqValid(const httplib::Request &request, const ReqCtxPtr &requestContext)
{
    std::string reqError = "";
    if (GetInferInstance()->GetPDRoleStatus() == PDRoleStatus::UNKNOWN) {
        reqError = "The server cannot process the inference request due to an unknown status. ";
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            CHECK_ERROR), "The server cannot process the inference request due to an unknown status.");
        HttpRestResource::ResponseJsonBody(requestContext, httplib::StatusCode::ServiceUnavailable_503,
            HttpRestResource::WrapperJson(reqError,
            g_exceptionInfo.at(httplib::StatusCode::ServiceUnavailable_503)));
        return false;
    }
    if (!(IsPrefillRole(reqError) &&
        IsAllDMIHeadersExist(request, reqError) &&
        IsReqIdValid(request, reqError) &&
        IsReqTypeValid(request, reqError) &&
        IsDTargetValid(request, reqError) &&
        IsRecomputeParamValid(request, reqError))) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            CHECK_ERROR), "DMI request invalid");
        HttpRestResource::ResponseJsonBody(requestContext, httplib::StatusCode::BadRequest_400,
            HttpRestResource::WrapperJson(reqError, g_exceptionInfo.at(httplib::StatusCode::BadRequest_400)));
        return false;
    }
    return true;
}

bool HttpHandler::CheckDresultReqValid(const ReqCtxPtr &requestContext)
{
    std::string reqError = "";
    if (!IsDMI()) {
        reqError = "Non DMI does not support dresult request";
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            CHECK_ERROR), reqError);
        HttpRestResource::ResponseJsonBody(requestContext, httplib::StatusCode::BadRequest_400,
            HttpRestResource::WrapperJson(reqError, g_exceptionInfo.at(httplib::StatusCode::BadRequest_400)));
        return false;
    }
    std::string pdRole = GetInferInstance()->GetPDRole();
    if (pdRole != "decode" && pdRole != "flex") {
        reqError = "Only the Decode node supports dresult but you send to ";
        reqError += pdRole;
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            CHECK_ERROR), reqError);
        HttpRestResource::ResponseJsonBody(requestContext, httplib::StatusCode::BadRequest_400,
            HttpRestResource::WrapperJson(reqError, g_exceptionInfo.at(httplib::StatusCode::BadRequest_400)));
        return false;
    }
    return true;
}

void HttpHandler::InitializeMetricsResource(HttpsServerHelper &server)
{
    /// Self-developed query statistic info interface(Prometheus format)
    ///
    /// For more information please refer to MINDIE documentation.
    server.Get("/metrics", [](const httplib::Request &request, httplib::Response &response) {
        std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
        GetPrometheusMetrics(context);
        return;
    });
}
} // namespace mindie_llm