/**
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

#include "llm_manager_v2/llm_manager_v2.h"
#include "src/llm_manager_v2/include/impl/llm_manager_impl.h"
#include "llm_manager/llm_manager.h"
#include "src/server/endpoint/utils/parameters_checker.h"
#include "check_utils.h"
#include "memory_utils.h"
#include "system_log.h"

namespace mindie_llm {

void TransformInputId(std::shared_ptr<InferRequest>& req, std::shared_ptr<Request>& v2Req)
{
    TensorPtr inputIdTensorPtr = nullptr;
    req->GetTensorByName("INPUT_IDS", inputIdTensorPtr);
    if (inputIdTensorPtr == nullptr) {
        LOG_ERROR_LLM << "INPUT_IDS tensor not found in request";
        return;
    }
    int64_t* inputIdData = static_cast<int64_t*>(inputIdTensorPtr->GetData());
    if (inputIdData == nullptr) {
        LOG_ERROR_LLM << "INPUT_IDS tensor data is null";
        return;
    }
    v2Req->input_token_num = inputIdTensorPtr->GetShape()[1];
    for (int i = 0; i < inputIdTensorPtr->GetShape()[1]; i++) {
        v2Req->input_ids.push_back(inputIdData[i]);
    }
}

void TransformStopTokenIds(std::shared_ptr<InferRequest>& req, std::shared_ptr<Request>& v2Req)
{
    TensorPtr stopTokenIdsTensorPtr = nullptr;
    req->GetTensorByName("STOP_TOKEN_IDS", stopTokenIdsTensorPtr);
    if (stopTokenIdsTensorPtr == nullptr) {
        LOG_ERROR_LLM << "STOP_TOKEN_IDS tensor not found in request";
        return;
    }
    TokenId* stopTokenIdsTensorData = static_cast<TokenId*>(stopTokenIdsTensorPtr->GetData());
    if (stopTokenIdsTensorData == nullptr) {
        LOG_ERROR_LLM << "INCLUDE_STOP_STR_IN_OUTPUT tensor data is null";
        return;
    }
    for (int i = 0; i < stopTokenIdsTensorPtr->GetShape()[1]; i++) {
        v2Req->stopTokenIds.value().push_back(stopTokenIdsTensorData[i]);
    }
}

void TransformRequest(std::shared_ptr<InferRequest>& req, std::shared_ptr<Request>& v2Req)
{
    auto transform = [&](const std::string& name, auto processor) {
        TensorPtr tensor = nullptr;
        req->GetTensorByName(name.c_str(), tensor);
        if (tensor == nullptr || (tensor->GetData() == nullptr)) {
            LOG_ERROR_LLM << name << " tensor not found or data is null";
            return;
        }
        processor(tensor->GetData(), tensor);
    };

    auto scalar = [](auto& field) {
        return [&field](void* data, TensorPtr) {
            if constexpr (is_optional_v<std::remove_reference_t<decltype(field)>>) {
                using ValueType = typename std::remove_reference_t<decltype(field)>::value_type;
                field = static_cast<ValueType*>(data)[0];
            } else {
                field = static_cast<std::remove_reference_t<decltype(field)>*>(data)[0];
            }
        };
    };

    auto str = [](auto& field) {
        return [&field](void* data, TensorPtr) {
            field = std::string(static_cast<char*>(data));
        };
    };

    transform("LORA_ID", str(v2Req->loraId));
    transform("IGNORE_EOS", scalar(v2Req->ignoreEos));
    transform("STOP_STRINGS", str(v2Req->stopStrings));
    transform("LOGPROBS", scalar(v2Req->logprobs));
    transform("TOP_LOGPROBS", scalar(v2Req->topLogprobs));
    transform("TEMPERATURE", scalar(v2Req->temperature));
    transform("TOP_K", scalar(v2Req->topK));
    transform("TOP_P", scalar(v2Req->topP));
    transform("TYPICAL_P", scalar(v2Req->typicalP));
    transform("DO_SAMPLE", scalar(v2Req->doSample));
    transform("SEED", scalar(v2Req->seed));
    transform("REPETITION_PENALTY", scalar(v2Req->repetitionPenalty));
    transform("FREQUENCY_PENALTY", scalar(v2Req->frequencyPenalty));
    transform("PRESENCE_PENALTY", scalar(v2Req->presencyPenalty));
    transform("INCLUDE_STOP_STR_IN_OUTPUT", scalar(v2Req->includeStopStrInOutput));
    transform("WATERMARK", scalar(v2Req->watermark));
    transform("N", scalar(v2Req->n));
    transform("BEST_OF", scalar(v2Req->bestOf));
    transform("USE_BEAM_SEARCH", scalar(v2Req->useBeamSearch));
}

std::vector<std::shared_ptr<Request>> AdaptGetRequestV1ToV2(GetRequestsCallback getRequest)
{
    auto v1Requests = getRequest();
    std::vector<std::shared_ptr<Request>> v2Requests;
    for (auto &req : v1Requests) {
        std::shared_ptr<Request> v2Req = std::make_shared<Request>();
        v2Req->requestId = req->GetRequestId().GetRequestIdString();
        TransformInputId(req, v2Req);
        TransformStopTokenIds(req, v2Req);
        TransformRequest(req, v2Req);
        v2Requests.push_back(v2Req);
    }
    return v2Requests;
}

std::shared_ptr<InferTensor> TransformMetrics(std::shared_ptr<Response>& response)
{
    std::vector<uint64_t> metrics;
    metrics.emplace_back(response->metrics.batchSize);
    metrics.emplace_back(response->metrics.queueWaitTime);
    std::vector<int64_t> shape = {1, static_cast<int64_t>(metrics.size())};
    auto tensor = std::make_shared<InferTensor>("METRICS", InferDataType::TYPE_UINT64, shape);
    tensor->Allocate(metrics.size() * sizeof(uint64_t));
    auto *buffer = reinterpret_cast<uint64_t *>(tensor->GetData());
    tensor->SetRelease(false);
    for (std::size_t i = 0; i < metrics.size(); i++) {
        buffer[i] = metrics[i];
    }
    return tensor;
}

template<typename T, InferDataType DataType>
std::shared_ptr<InferTensor> CreateVectorTensor(const std::string& name, std::vector<T> value)
{
    std::vector<int64_t> shape = {1, static_cast<int64_t>(value.size())};
    auto tensor = std::make_shared<InferTensor>(name, DataType, shape);
    tensor->Allocate(value.size() * sizeof(T));
    std::copy(value.begin(), value.end(), static_cast<T *>(tensor->GetData()));
    return tensor;
}

void CreateScalarTensor(const std::string &tensorName, std::vector<size_t> tensorShape, InferDataType tensorType,
                        PVOID tensorData, TensorMap &tensors)
{
    // 计算所需空间
    size_t tensorSize = InferTensor::GetTypeByteSize(tensorType);
    bool overflowFlag = false;
    for (size_t dimSize : tensorShape) {
        tensorSize = IntMulWithCheckOverFlow(tensorSize, dimSize, overflowFlag);
        if (overflowFlag) {
            LOG_ERROR_LLM << "Overflow detected during AddTensorToResponse";
            return;
        }
    }
    // std::vector<size_t> => std::vector<int64_t>
    std::vector<int64_t> tensorShapeAsInt64{};
    for (size_t dimensionSize : tensorShape) {
        tensorShapeAsInt64.push_back(static_cast<int64_t>(dimensionSize));
    }
    auto responseTensor = std::make_shared<InferTensor>(tensorName, tensorType, tensorShapeAsInt64);
    auto ret = tensors.insert(std::make_pair(responseTensor->GetName(), responseTensor));
    if (!ret.second) {
        LOG_ERROR_LLM << "The tensor " << responseTensor->GetName() << " already exists!";
    }
    // 重新分配内存
    // 解决 infer engine 回调与 endpoint 层超时撞上，引起内存释放后使用问题, from commit 9446b0b
    if (!responseTensor->Allocate(tensorSize)) {
        tensors.erase(responseTensor->GetName());
        return;
    }
    if (tensorName == "METRICS") {
        responseTensor->SetRelease(false);
    }
    auto copyRet =
        memcpy_s(responseTensor->GetData(), responseTensor->GetSize(), tensorData, responseTensor->GetSize());
    if (copyRet != 0) {
        tensors.erase(responseTensor->GetName());
        throw std::runtime_error("Memory copy for tensor " + tensorName +
                                     " failed!, tensor size=" + std::to_string(tensorSize));
    }
}

struct TensorData {
    std::vector<SequenceId> seqIds;              // (seq_num,)
    std::vector<SequenceId> parentSeqIds;        // (seq_num,)
    std::vector<TokenId> tokenIds;               // (seq_num, gen_token_num)
    std::vector<Probability> logProbs;           // (seq_num, gen_token_num)
    std::vector<int64_t> eosAttr;                // (seq_num, 2)
    std::vector<int64_t> truncationIndex;        // (seq_num,)
    std::vector<TokenId> topTokenIds;            // (seq_num, top_k, gen_token_num)
    std::vector<Probability> topLogProbs;        // (seq_num, top_k, gen_token_num)
    std::vector<Probability> cumulativeLogProbs; // (seq_num,)
};

void TransformOutPut(TensorMap& tensors, std::shared_ptr<Response>& response)
{
    TensorData tensorData;
    for (ResponseContent content :response->responseContents) {
        tensorData.seqIds.push_back(content.seqId);
        tensorData.parentSeqIds.push_back(content.parentSeqId);
        tensorData.eosAttr.push_back(static_cast<int64_t>(content.finishReason));
        tensorData.eosAttr.push_back(content.speculativeTokenNum);
        tensorData.truncationIndex.push_back(content.truncationIndex);
        tensorData.cumulativeLogProbs.push_back(content.cumLogProb);

        std::copy(content.outTokenIds.begin(), content.outTokenIds.end(),
                  std::back_inserter(tensorData.tokenIds));
        std::copy(content.outLogProbs.begin(), content.outLogProbs.end(), std::back_inserter(tensorData.logProbs));

        std::copy(content.topLogProbTokenIds.begin(), content.topLogProbTokenIds.end(),
                  std::back_inserter(tensorData.topTokenIds));
        std::copy(content.topLogProbs.begin(), content.topLogProbs.end(),
                  std::back_inserter(tensorData.topLogProbs));
    }
    size_t seqNum = tensorData.seqIds.size();
    if (seqNum == 0) {
        throw std::runtime_error("[LlmManagerAdapter|TransformOutPut] seqNum can not be 0");
    }
    size_t tokenNum = tensorData.tokenIds.size() / seqNum;
    size_t numParallelTokens = static_cast<size_t>(response->numParallelTokens);
    if (numParallelTokens == 0) {
        throw std::runtime_error("numParallelTokens can not be 0");
    }
    size_t numTopTokens = tensorData.topTokenIds.size() / (seqNum * numParallelTokens);

    CreateScalarTensor("IBIS_SEQS_ID", {seqNum}, InferDataType::TYPE_INT64,
        tensorData.seqIds.data(), tensors);
    CreateScalarTensor("PARENT_SEQS_ID", {seqNum}, InferDataType::TYPE_INT64,
        tensorData.parentSeqIds.data(), tensors);
    CreateScalarTensor("OUTPUT_IDS", {seqNum, tokenNum}, InferDataType::TYPE_INT64,
        tensorData.tokenIds.data(), tensors);
    CreateScalarTensor("OUTPUT_LOGPROBS", {seqNum, tokenNum}, InferDataType::TYPE_FP32,
        tensorData.logProbs.data(), tensors);
    CreateScalarTensor("IBIS_EOS_ATTR", {seqNum, 2}, InferDataType::TYPE_INT64,
        tensorData.eosAttr.data(), tensors);
    CreateScalarTensor("TRUNCATION_INDICES", {seqNum}, InferDataType::TYPE_INT64,
        tensorData.truncationIndex.data(), tensors);
    CreateScalarTensor("TOP_TOKEN_IDS", {seqNum, numParallelTokens, numTopTokens},
        InferDataType::TYPE_INT64, tensorData.topTokenIds.data(), tensors);
    CreateScalarTensor("TOP_LOGPROBS", {seqNum, numParallelTokens, numTopTokens},
        InferDataType::TYPE_FP32, tensorData.topLogProbs.data(), tensors);
    CreateScalarTensor("CUMULATIVE_LOGPROBS", {seqNum}, InferDataType::TYPE_FP32,
        tensorData.cumulativeLogProbs.data(), tensors);
}

void AdaptSendResponseV2ToV1(SendResponsesCallback sendResponse, std::shared_ptr<Response> response)
{
    InferRequestId requestId(response->reqId);
    TensorMap tensors;
    bool isFinal = response->isEos;
    std::string errorMsg;
    tensors["METRICS"] = TransformMetrics(response);
    if (!response->responseContents.empty()) {
        TransformOutPut(tensors, response);
    }
    errorMsg = std::to_string(static_cast<int>(response->inferStatusFlag));
    sendResponse(requestId, tensors, isFinal, errorMsg);
}

std::vector<std::pair<RequestIdNew, OperationV2>> AdaptControlSignalCallbackV1ToV2(
    ControlSignalCallback controlCallback)
{
    std::vector<std::pair<RequestIdNew, OperationV2>> resultsV2;
    std::vector<std::pair<InferRequestId, Operation>> resultsV1 = controlCallback();
    for (auto &item : resultsV1) {
        InferRequestId requestId = item.first;
        Operation operation = item.second;

        // 类型转换
        RequestIdNew newRequestId(requestId.StringValue());
        OperationV2 newOperation = static_cast<OperationV2>(operation);

        // 添加到结果向量
        resultsV2.emplace_back(newRequestId, newOperation);
    }
    return resultsV2;
}

void AdaptStatusResponseCallbackV2ToV1(SendStatusResponseCallback statusResponseCallback, RequestIdNew requestId,
                                       Status status, StatusResponseTypeV2 statusType)
{
    InferRequestId requestIdV1(requestId);
    statusResponseCallback(requestIdV1, status, static_cast<StatusResponseType>(statusType));
}

} // namespace mindie_llm