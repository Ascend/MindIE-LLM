/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef OCK_ENDPOINT_HTTP_HANDLER_H
#define OCK_ENDPOINT_HTTP_HANDLER_H
#define CPPHTTPLIB_OPENSSL_SUPPORT

#include <sys/wait.h>
#include <thread>
#include <mutex>
#include <memory>
#include <atomic>

#include "httplib.h"
#include "config_manager.h"
#include "http_rest_resource.h"
#include "single_llm_pnd_req_handler.h"
#include "https_server_helper.h"
#include "single_req_infer_interface_base.h"
#include "request_response/response.h"
#include "health_checker.h"

using ordered_json = nlohmann::ordered_json;

namespace mindie_llm {
class HttpHandler {
public:
    static int BusinessInitialize(HttpsServerHelper &server);
    static int ManagementInitialize(HttpsServerHelper &server);
    static void InitializeMetricsResource(HttpsServerHelper &server);
    static bool JudgeRestProcess();

private:
    static void InitializeServiceStatusResource(HttpsServerHelper &server);
    static void RegisterHealthCheckerHttpHandler(HttpsServerHelper &server);
    static void InitializeServiceMgmtV1(HttpsServerHelper &server);
    static void InitializeServiceMgmtV2(HttpsServerHelper &server);
    static void InitializeManagerResource(HttpsServerHelper &server);
    static void InitializeManagerResourceV1(HttpsServerHelper &server);
    static void InitializeManagerResourceV2(HttpsServerHelper &server);
    static void InitializeInferResource(HttpsServerHelper &server);
    static void InitializeInferResourceV2(HttpsServerHelper &server);

    static bool GetRequestModelConfig(const ReqCtxPtr &requestContext, ModelDeployConfig &config);
    static bool GetRequestModelConfig(const ReqCtxPtr &requestContext, LoraConfig &config);
    static bool HandlePostGenerate(ReqCtxPtr &reqCtx);
    static int HandleGetSlotCount(const std::shared_ptr<RequestContext> &context);
    static void HandleGetHealthStatus(const ReqCtxPtr &ctx);
    static void HandlePDWiseUpdateNpuDeviceIds(const ReqCtxPtr &ctx);
    static void HandleUpdateNpuDeviceIds(const ReqCtxPtr &ctx);
    static void HandleTokenizer(const ReqCtxPtr &ctx);
    static void HandleStatusV1(const ReqCtxPtr &ctx);
    static void HandleStatusV2(const ReqCtxPtr &ctx);
    static void HandleGeneralTGIPostGenerate(const httplib::Request &request, ReqCtxPtr &reqCtx);
    static void HandleGetLivenessAndReadiness(const ReqCtxPtr &requestContext);
    static void HandleGetEngineStatus(const ReqCtxPtr &requestContext);
    static void HandlePostCmdToEngine(const ReqCtxPtr &reqCtx);
    static void ExecuteFaultRecoveryPauseCmd(RecoverCommandInfo &info, Status &status);
    static void ExecuteFaultRecoveryReinitNpuCmd(RecoverCommandInfo &info, Status &status);
    static void ExecuteFaultRecoveryStartEngineCmd(RecoverCommandInfo &info, Status &status);
    static void ForceStop(const ReqCtxPtr &requestContext);
    static void StopService(const ReqCtxPtr &requestContext);
    static void HandleHttpMetrics(const ReqCtxPtr &requestContext);
    static void HandleHealthResponse(const ReqCtxPtr &ctx, mindie_llm::ServiceStatus status, bool isLiveness);
    static void RespondUnhealthy(const ReqCtxPtr &ctx, const std::string& message);
    static bool IsDMI();
    static bool IsPrefillRole(std::string &reqError);
    static bool IsAllDMIHeadersExist(const httplib::Request &request, std::string &reqError);
    static bool IsReqIdValid(const httplib::Request &request, std::string &reqError);
    static bool IsReqTypeValid(const httplib::Request &request, std::string &reqError);
    static bool IsDTargetValid(const httplib::Request &request, std::string &reqError);
    static bool IsRecomputeParamValid(const httplib::Request &request, std::string &reqError);
    static bool CheckDMIReqValid(const httplib::Request &request, const ReqCtxPtr &requestContext);
    static bool CheckDresultReqValid(const ReqCtxPtr &requestContext);
    static void GetPrometheusMetrics(const ReqCtxPtr &requestContext);

private:
    static void SetJsonObj(ordered_json &jsonObj);

    static int64_t startTime;
    static std::mutex dmiRoleMutex_;  // already update
};
} // namespace mindie_llm

#endif // OCK_ENDPOINT_HTTP_HANDLER_H
