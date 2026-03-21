/**
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

#include "log.h"
#include "dmi_msg_receiver.h"

namespace mindie_llm {
        grpc::Status DecodeRequestReceiver::DecodeRequestChannel([[maybe_unused]] grpc::ServerContext* context,
            const prefillAndDecodeCommunication::DecodeParameters* request,
            prefillAndDecodeCommunication::DecodeRequestResponse* response)
        {
            (void)context;

            ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Get decode request success, requestId: " << request->reqid());
            std::string errMsg = "";
            if (!isValidRequest(request, response, errMsg)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "Request or Response is invalid");
                if (response != nullptr) {
                    response->set_isvaliddecodeparameters(false);
                    response->set_errormessage(errMsg);
                }
                return grpc::Status::CANCELLED;
            }
            if (decodeRequestHandler_ == nullptr) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "The decodeRequestHandler_ is invalid");
                return grpc::Status::CANCELLED;
            }
            decodeRequestHandler_(*request, *response);
            return grpc::Status::OK;
        }

        bool DecodeRequestReceiver::RegisterMsgHandler(DecodeRequestHandler callback)
        {
            if (callback == nullptr) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "The callback is nullptr");
                return false;
            }
            decodeRequestHandler_ = std::move(callback);
            return true;
        }

        bool DecodeRequestReceiver::isValidRequest(const prefillAndDecodeCommunication::DecodeParameters* request,
                                                   prefillAndDecodeCommunication::DecodeRequestResponse* response,
                                                   std::string& errMsg)
        {
            if (response == nullptr) {
                errMsg = "Response is nullptr";
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), errMsg);
                return false;
            }

            if (request == nullptr) {
                errMsg = "Request is nullptr";
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), errMsg);
                return false;
            }

            if (request->maxnewtoken() < 0) {
                errMsg = "MaxOutPutLen is invalid";
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), errMsg);
                return false;
            }

            return true;
        }

        grpc::Status KvReleaseReceiver::ReleaseKVCacheChannel([[maybe_unused]] grpc::ServerContext* context,
            const prefillAndDecodeCommunication::RequestId* request,
            [[maybe_unused]] google::protobuf::Empty* response)
        {
            (void)context;
            (void)response;
            ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Get kv release request success, requestId: " << request->reqid());

            if (!isValidRequest(request)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "Request is invalid");
                return grpc::Status::CANCELLED;
            }
            if (kvReleaseHandler_ == nullptr) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "The kvReleaseHandler_ is invalid");
                return grpc::Status::CANCELLED;
            }
            auto reqId = request->reqid();
            std::thread([this, reqId]() {
                pthread_setname_np(pthread_self(), "GRPCRequest");
                // 执行请求处理逻辑
                this->kvReleaseHandler_(reqId);
            }).detach();
            return grpc::Status::OK;
        }

        bool KvReleaseReceiver::RegisterMsgHandler(KVReleaseHandler callback)
        {
            if (callback == nullptr) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "The callback is nullptr");
                return false;
            }
            kvReleaseHandler_ = std::move(callback);
            return true;
        }

        bool KvReleaseReceiver::isValidRequest(const prefillAndDecodeCommunication::RequestId* request)
        {
            if (request == nullptr) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "The request is nullptr");
                return false;
            }
            return true;
        }

} // namespace mindie_llm
