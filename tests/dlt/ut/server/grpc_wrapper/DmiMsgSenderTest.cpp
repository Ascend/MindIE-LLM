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
#include <gtest/gtest.h>
#include "mockcpp/mockcpp.hpp"
#include <string>
#include <thread>
#include "dmi_msg_sender.h"
#include <grpcpp/channel.h>
#include <grpcpp/create_channel.h>

using namespace prefillAndDecodeCommunication;

namespace grpc {
bool operator==(const Status& lhs, const Status& rhs)
{
    return lhs.error_code() == rhs.error_code() &&
           lhs.error_message() == rhs.error_message();
}

bool operator!=(const Status& lhs, const Status& rhs)
{
    return !(lhs == rhs);
}
}

namespace mindie_llm {

class DmiMsgSenderTest : public testing::Test {
protected:
    void SetUp() override
    {
    }

    void TearDown() override
    {
        GlobalMockObject::verify();
    }

};

TEST_F(DmiMsgSenderTest, DecodeRequestSender_InitWithoutTlsSuccess)
{
    DecodeRequestSender sender("127.0.0.1:50051", "127.0.0.1:50052", false, nullptr);
    EXPECT_TRUE(sender.Init());
}

TEST_F(DmiMsgSenderTest, DecodeRequestSender_InitWithTlsSuccess)
{
    auto tlsOptions = std::make_unique<grpc::experimental::TlsChannelCredentialsOptions>();
    DecodeRequestSender sender("127.0.0.1:50051", "127.0.0.1:50052", true, std::move(tlsOptions));
    EXPECT_TRUE(sender.Init());
}

TEST_F(DmiMsgSenderTest, DecodeRequestSender_InitWithTlsButNoOptions)
{
    DecodeRequestSender sender("127.0.0.1:50051", "127.0.0.1:50052", true, nullptr);
    EXPECT_FALSE(sender.Init());
}

TEST_F(DmiMsgSenderTest, DecodeRequestSender_CreateStubSuccess)
{
    auto channel = grpc::CreateChannel("127.0.0.1:50051", grpc::InsecureChannelCredentials());
    DecodeRequestSender sender("127.0.0.1:50051", "127.0.0.1:50052", false, nullptr);
    sender.CreateStub(channel);
    EXPECT_EQ(sender.receiverAddr_, "127.0.0.1:50051");
    EXPECT_EQ(sender.localAddr_, "127.0.0.1:50052");
}

TEST_F(DmiMsgSenderTest, DecodeRequestSender_ResponseIsNotValiddecodeParameters)
{
    auto channel = grpc::CreateChannel("127.0.0.1:50051", grpc::InsecureChannelCredentials());
    DecodeRequestSender sender("127.0.0.1:50051", "127.0.0.1:50052", false, nullptr);
    sender.CreateStub(channel);
    MOCKER_CPP(&DecodeService::Stub::DecodeRequestChannel, grpc::Status (*)(
                grpc::ClientContext*, const DecodeParameters&, DecodeRequestResponse*))
                .stubs().will(returnValue(grpc::Status::OK));
    std::string errMsg;
    DecodeParameters request;
    std::string reqId = "req-";
    EXPECT_FALSE(sender.SendDecodeRequestMsg(request, reqId, errMsg));
    EXPECT_EQ(errMsg, "");
}

TEST_F(DmiMsgSenderTest, DecodeRequestSender_StubNull)
{
    DecodeRequestSender sender("127.0.0.1:50051", "127.0.0.1:50052", false, nullptr);
    MOCKER_CPP(&DecodeService::Stub::DecodeRequestChannel, grpc::Status (*)(
                grpc::ClientContext*, const DecodeParameters&, DecodeRequestResponse*))
                .stubs().will(returnValue(grpc::Status::OK));
    std::string errMsg;
    DecodeParameters request;
    std::string reqId = "req-";
    EXPECT_FALSE(sender.SendDecodeRequestMsg(request, reqId, errMsg));
    EXPECT_EQ(errMsg, "The stub_ is nullptr");
}

TEST_F(DmiMsgSenderTest, DecodeRequestSender_DecodeRequestChannelError)
{
    auto channel = grpc::CreateChannel("127.0.0.1:50051", grpc::InsecureChannelCredentials());
    DecodeRequestSender sender("127.0.0.1:50051", "127.0.0.1:50052", false, nullptr);
    sender.CreateStub(channel);
    MOCKER_CPP(&DecodeService::Stub::DecodeRequestChannel, grpc::Status (*)(
                grpc::ClientContext*, const DecodeParameters&, DecodeRequestResponse*))
                .stubs().will(returnValue(grpc::Status::CANCELLED));
    std::string errMsg;
    DecodeParameters request;
    std::string reqId = "req-";
    EXPECT_FALSE(sender.SendDecodeRequestMsg(request, reqId, errMsg));
    EXPECT_EQ(errMsg, "Failed to send decode request msg because[1]  receiverAddr is 127.0.0.1:50051. RequestId is req-");
}

TEST_F(DmiMsgSenderTest, KvReleaseSender_CreateStubSuccess)
{
    auto channel = grpc::CreateChannel("127.0.0.1:50051", grpc::InsecureChannelCredentials());
    KvReleaseSender sender("127.0.0.1:50051", "127.0.0.1:50052", false, nullptr);
    sender.CreateStub(channel);
    EXPECT_EQ(sender.receiverAddr_, "127.0.0.1:50051");
    EXPECT_EQ(sender.localAddr_, "127.0.0.1:50052");
}

TEST_F(DmiMsgSenderTest, KvReleaseSender_Success)
{
    auto channel = grpc::CreateChannel("127.0.0.1:50051", grpc::InsecureChannelCredentials());
    KvReleaseSender sender("127.0.0.1:50051", "127.0.0.1:50052", false, nullptr);
    sender.CreateStub(channel);
    MOCKER_CPP(&PrefillService::Stub::ReleaseKVCacheChannel, grpc::Status (*)(
                grpc::ClientContext*, const RequestId&, google::protobuf::Empty*))
                .stubs().will(returnValue(grpc::Status::OK));
    RequestId request;
    EXPECT_TRUE(sender.SendKvReleaseMsg(request));
}

TEST_F(DmiMsgSenderTest, KvReleaseSender_StubNull)
{
    KvReleaseSender sender("127.0.0.1:50051", "127.0.0.1:50052", false, nullptr);
    MOCKER_CPP(&PrefillService::Stub::ReleaseKVCacheChannel, grpc::Status (*)(
                grpc::ClientContext*, const RequestId&, google::protobuf::Empty*))
                .stubs().will(returnValue(grpc::Status::OK));
    RequestId request;
    std::string reqId = "req-";
    EXPECT_FALSE(sender.SendKvReleaseMsg(request));
}

TEST_F(DmiMsgSenderTest, KvReleaseSender_DecodeRequestChannelError)
{
    auto channel = grpc::CreateChannel("127.0.0.1:50051", grpc::InsecureChannelCredentials());
    KvReleaseSender sender("127.0.0.1:50051", "127.0.0.1:50052", false, nullptr);
    sender.CreateStub(channel);
    MOCKER_CPP(&PrefillService::Stub::ReleaseKVCacheChannel, grpc::Status (*)(
                grpc::ClientContext*, const RequestId&, google::protobuf::Empty*))
                .stubs().will(returnValue(grpc::Status::CANCELLED));
    RequestId request;
    EXPECT_FALSE(sender.SendKvReleaseMsg(request));
}

} // namespace mindie_llm
