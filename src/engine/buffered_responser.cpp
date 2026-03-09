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
#include "buffered_responser.h"
#include "system_log.h"

using namespace std;
using namespace mindie_llm;

BufferedResponser::BufferedResponser(ForwardRespToManagerCall cb, BufferResponseConfig &config)
    : forwardRespToManagerCall_(cb), bufferResponseConfig_(config)
{
    if (bufferResponseConfig_.bufferResponseEnabled) {
        respBufferThread_ = std::thread([this]() {
            pthread_setname_np(pthread_self(), "RespBufferThread");
            RespBufferThread();
        });
    }
}

BufferedResponser::~BufferedResponser()
{
    if (bufferResponseConfig_.bufferResponseEnabled) {
        stop_.store(true, memory_order_relaxed);
        if (respBufferThread_.joinable()) {
            respBufferThread_.join();
        }
    }
}

void BufferedResponser::RespBufferThread()
{
    double prefillExpectedTime = bufferResponseConfig_.prefillExpectedTime;
    double decodeExpectedTime = bufferResponseConfig_.decodeExpectedTime;
    // 给SLO时延留有余量，如果超过0.95x时限就发出去
    while (!stop_) {
        std::vector<string> allReqIds = respBufferMap_.KeySet();
        for (const string& reqId : allReqIds) {
            std::optional<ResponseBufferPtr> optMetadata = respBufferMap_.Get(reqId);
            if (!optMetadata.has_value()) {
                continue;
            }
            ResponseBufferPtr responseBuffer = optMetadata.value();
            if (responseBuffer->IsEnded()) {
                SendEndResponse(responseBuffer);
                respBufferMap_.Erase(reqId);  // 从ConcurrentMap中删除
            } else {
                MaybeSendContinueResponse(responseBuffer, prefillExpectedTime, decodeExpectedTime);
            }
        }
        // 避免频繁轮询导致CPU飚高，sleep 1毫秒
        this_thread::sleep_for(chrono::milliseconds(1));
    }
}

void BufferedResponser::TryRespond(const ResponseSPtr &response)
{
    string reqId = response->reqId;
    std::optional<ResponseBufferPtr> bufferOpt = respBufferMap_.Get(reqId);
    if (!bufferOpt.has_value()) {
        LOG_DEBUG_LLM << "No buffer found for request: " << reqId;
        return;
    }
    ResponseBufferPtr responseBuffer = bufferOpt.value();
    // 当prefill请求返回eos token时, 直接发送response
    if (responseBuffer->GetInferStage() == InferReqType::REQ_PREFILL && response->isEos
        && responseBuffer->IsEmpty()) {
        forwardRespToManagerCall_(response);
        respBufferMap_.Erase(reqId);
        return;
    }
    responseBuffer->AddResponse(response);
    if (response->isEos) {
        responseBuffer->SetReqEnded();
    }
}

void BufferedResponser::RecordArriveTime(RequestIdNew inferReqId,
    chrono::time_point<chrono::high_resolution_clock> arriveTime)
{
    if (!bufferResponseConfig_.bufferResponseEnabled) {
        return;
    }

    int64_t reqArrivalTime =
        chrono::time_point_cast<chrono::nanoseconds>(arriveTime).time_since_epoch().count();
    // 只记录prefill的到达时间
    if (respBufferMap_.Count(inferReqId) == 0) {
        std::shared_ptr<ResponseBuffer> metadata = make_shared<ResponseBuffer>(
            InferReqType::REQ_PREFILL, reqArrivalTime
        );
        respBufferMap_.Insert(inferReqId, metadata);
    }
}

void BufferedResponser::SendEndResponse(ResponseBufferPtr &responseBuffer)
{
    // 识别到已经结束，立即把前面的响应一并返回，并把响应从buffer map里清理掉
    while (!responseBuffer->IsEmpty()) {
        ResponseSPtr response = responseBuffer->PopFront();
        if (response) { // 检查是否成功获取数据
            forwardRespToManagerCall_(response);
        }
    }
}

void BufferedResponser::MaybeSendContinueResponse(
    ResponseBufferPtr &responseBuffer, double prefillExpectedTime, double decodeExpectedTime)
{
    if (responseBuffer->IsEmpty()) {
        return;
    }
    // 如果没有结束，卡在SLO之前响应
    int64_t curTime = chrono::time_point_cast<chrono::nanoseconds>(chrono::high_resolution_clock::now())
        .time_since_epoch().count();
    double diffTime = static_cast<double>(
        (curTime - responseBuffer->GetlastRespArriveTime()) / changeNsToMs);
    double sloExpectedTime;
    if (responseBuffer->GetInferStage() == InferReqType::REQ_PREFILL) {
        sloExpectedTime = prefillExpectedTime * sloBufferRatio;
    } else {
        sloExpectedTime = decodeExpectedTime * sloBufferRatio;
    }
    // 给SLO时延留有余量，如果超过期望时间就发出去
    if (diffTime >= sloExpectedTime) {
        ResponseSPtr response = responseBuffer->PopFront();
        if (response) { // 检查是否成功获取数据
            forwardRespToManagerCall_(response);
        }
        responseBuffer->SetlastRespArriveTime(curTime);
        responseBuffer->SetInferStage(InferReqType::REQ_DECODE);
    }
}