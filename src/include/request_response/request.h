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
 
#ifndef MINDIE_LLM_REQUEST_H
#define MINDIE_LLM_REQUEST_H

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <map>
#include <functional>
#include "basic_types.h"
#include "data_type.h"
#include "pd_role.h"
#include "request_id.h"
#include "response.h"
#include "request_response/callback.h"

namespace mindie_llm {
struct FailedLinkInfo {
    uint64_t cluster_id;
    int64_t failReason;
};

struct Request {
    Request() = default;
    explicit Request(const RequestIdNew &rid) : requestId(rid) {}

    RequestIdNew requestId;
    int64_t input_token_num = 0;
    std::vector<int64_t> input_ids = {};
    bool isSynchronous = false;
    uint64_t maxOutputLen = 20; // MAX_NEW_TOKENS_DFT
    std::optional<bool> ignoreEos;
    uint32_t windowSize = 0;
    std::optional<float> temperature;
    std::optional<int32_t> topK;
    std::optional<float> topP;
    std::optional<float> typicalP;
    std::optional<bool> doSample;
    std::optional<uint64_t> seed;
    std::optional<float> repetitionPenalty;
    std::optional<bool> watermark;
    std::optional<float> frequencyPenalty;
    std::optional<float> presencyPenalty;
    std::optional<std::vector<TokenId>> stopTokenIds;
    std::optional<std::string> stopStrings;
    std::optional<std::vector<std::string>> stopStrList;
    std::optional<bool> includeStopStrInOutput;
    std::optional<bool> skipSpecialTokens;
    std::optional<uint32_t> bestOf;
    std::optional<uint32_t> n;
    std::optional<bool> useBeamSearch;
    std::optional<uint32_t> topLogprobs;
    std::optional<bool> logprobs;
    std::optional<std::string> responseFormat; // JSON structured output format
    uint64_t priority = 5; // PRIORITY_DFT
    std::string loraId = "None";

    // For PD分离
    PDRole role = PDRole::UNKNOWN;
    bool needSwitch = false;
    uint32_t linkNum = 0;
    uint32_t unlinkNum = 0;
    uint32_t hostIpNum = 0;
    uint32_t superPodIdNum = 0;
    uint32_t containsDpInstanceIds = 0;
    std::vector<FailedLinkInfo> failedLinkInfos;
    // key: dp instance id -> value: sp size / cp size
    std::map<uint64_t, int64_t> spInfo{};
    std::map<uint64_t, int64_t> cpInfo{};

    InferReqType reqType = InferReqType::REQ_STAND_INFER;
    bool isSimulateRequest = false;  //< 是否为虚推请求
    bool isRecompute = false;
    std::optional<InstanceId> pInstanceId; // pull kv will use management port (pInstanceId)
    std::vector<std::vector<int64_t>> srcBlockTable; // block table from prefill
    std::vector<uint64_t> dpInstanceIds; // dp instance ids from prefill [maybe unused]
    // For Link/Unlink
    // {dpInstanceId: [host_ip1, host_ip2, ...]}
    std::unordered_map<InstanceId, std::vector<std::string>> dpInstance2HostIps;
    // {dpInstanceId: host_superPodId} superPodId is used for A3 machine
    std::unordered_map<InstanceId, int64_t> dpInstance2SuperPodId;
    // {dpInstanceId: [(device_ip1, device_physical_id1), ...]}
    std::unordered_map<InstanceId, std::vector<std::pair<std::string, int64_t>>> dpInstance2LinkDevices;
    // {dpInstanceId: [(device_ip1, device_physical_id1), ...]}
    std::unordered_map<InstanceId, std::vector<std::pair<std::string, int64_t>>> dpInstance2UnlinkDevices;
    // {dpInstanceId: [super_device_id1, super_device_id2, ...]}
    std::unordered_map<InstanceId, std::vector<int64_t>> dpInstance2LinkSuperDeviceIds;
    // {dpInstanceId: [super_device_id1, super_device_id2, ...]}
    std::unordered_map<InstanceId, std::vector<int64_t>> dpInstance2UnLinkSuperDeviceIds;
    SendResponsesCallbackV2 serverResponseCallback_{};

    bool HasStopWords()
    {
        return (stopStrings.has_value() && !stopStrings.value().empty())
            || (stopTokenIds.has_value() && !stopTokenIds.value().empty());
    }
};
using RequestSPtr = std::shared_ptr<Request>;

} // namespace mindie_llm

#endif
