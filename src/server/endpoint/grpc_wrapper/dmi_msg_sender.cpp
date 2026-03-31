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
 
#include <memory>
#include <sstream>
#include <grpcpp/grpcpp.h>
#include <grpcpp/channel.h>
#include <grpcpp/create_channel.h>

#include "log.h"
#include "health_checker/health_checker.h"
#include "dmi_msg_sender.h"

namespace mindie_llm {
        constexpr uint32_t DECODE_TIME_OUT = 30000;
        constexpr uint32_t KV_RELEASE_TIME_OUT = 60000;
        bool GrpcMsgSender::Init()
        {
            std::shared_ptr<grpc::Channel> channel;
            if (useTls_) {
                ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Client run with tls.");
                if (TlsChannelOpt_ == nullptr) {
                    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                        CHECK_ERROR), "Client ssl opt is nullptr");
                    return false;
                }
                // 获取存根
                std::shared_ptr<grpc::ChannelCredentials> creds = grpc::experimental::TlsCredentials(*TlsChannelOpt_);
                TlsChannelOpt_ = nullptr;
                channel = grpc::CreateChannel(receiverAddr_, creds);
            } else {
                ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Client init without tls.");
                channel = grpc::CreateChannel(receiverAddr_, grpc::InsecureChannelCredentials());
            }
            CreateStub(channel);
            return true;
        }

        void DecodeRequestSender::CreateStub(std::shared_ptr<grpc::Channel>& channel)
        {
            stub_ = std::move(prefillAndDecodeCommunication::DecodeService::NewStub(channel));
        }

        bool DecodeRequestSender::SendDecodeRequestMsg(const prefillAndDecodeCommunication::DecodeParameters &message,
            const std::string& reqId, std::string& errMsg)
        {
            std::unique_ptr<SendingDecodeMessageScope> sendingDecodeScope;
            if (HealthChecker::GetInstance().IsEnabled()) {
                sendingDecodeScope =
                    std::make_unique<SendingDecodeMessageScope>(HealthChecker::GetInstance());
            }
            {
                std::unique_lock <std::mutex> lock(lock_);
                // 每次发送时配置为 "Wait-for-Ready"
                grpc::ClientContext context;
                context.set_wait_for_ready(true);
                context.set_deadline(CalDeadLine(DECODE_TIME_OUT));
                prefillAndDecodeCommunication::DecodeRequestResponse response;
                if (stub_ == nullptr) {
                    errMsg = "The stub_ is nullptr";
                    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                        CHECK_ERROR), errMsg);
                    return false;
                }
                grpc::Status status = stub_->DecodeRequestChannel(&context, message, &response);
                if (!status.ok()) {
                    std::stringstream ss;
                    ss << "Failed to send decode request msg because[" << status.error_code() << "] "
                        << status.error_message() << " receiverAddr is " << receiverAddr_ << ". RequestId is " << reqId;
                    errMsg = ss.str();
                    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                        ABNORMAL_TRANSMISSION_ERROR), errMsg);
                    return false;
                }
                if (!response.isvaliddecodeparameters()) {
                    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                        CHECK_ERROR), "Decode node check para failed because " << response.errormessage());
                    return false;
                }
            }

            static uint64_t sendCnt = 0;
            return true;
        }

        void KvReleaseSender::CreateStub(std::shared_ptr<grpc::Channel>& channel)
        {
            stub_ = std::move(prefillAndDecodeCommunication::PrefillService::NewStub(channel));
        }

        bool KvReleaseSender::SendKvReleaseMsg(const prefillAndDecodeCommunication::RequestId &message)
        {
            std::unique_lock<std::mutex> lock(lock_);
            grpc::ClientContext context;
            // 设置超时为10秒
            context.set_deadline(CalDeadLine(KV_RELEASE_TIME_OUT));
            // 配置 "Wait-for-Ready"
            context.set_wait_for_ready(true);
            google::protobuf::Empty response;
            if (stub_ == nullptr) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "The stub_ is nullptr");
                return false;
            }
            grpc::Status status = stub_->ReleaseKVCacheChannel(&context, message, &response);
            if (!status.ok()) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    ABNORMAL_TRANSMISSION_ERROR), "Failed to send kv release msg because [" << status.error_code()
                    << "] " << status.error_message() << " receiverAddr is " << receiverAddr_ << ". requestId: "
                    << message.reqid());
                return false;
            }
            static uint64_t sendCnt = 0;
            return true;
        }

} // namespace mindie_llm
