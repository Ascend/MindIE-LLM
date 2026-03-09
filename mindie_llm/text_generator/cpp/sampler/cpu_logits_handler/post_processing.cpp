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

#include "post_processing.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <cfloat>
#include <map>
#include <deque>
#include "check_utils.h"
#include "system_log.h"

namespace {

const int MAX_SCORE_SIZE = std::numeric_limits<int>::max();

union UInt32Float {
    uint32_t uint32Value;
    float float32Value;

    explicit UInt32Float(uint32_t uint32Value) : uint32Value(uint32Value) {}
    explicit UInt32Float(float float32Value) : float32Value(float32Value) {}
};

inline float DecodeFp16(uint16_t float16Value)
{
    uint32_t sign = float16Value >> 15;
    uint32_t exponent = (float16Value >> 10) & 0x1F;
    uint32_t fraction = (float16Value & 0x3FF);
    UInt32Float uint32FloatValue(0.0f);
    if (exponent == 0) {
        if (fraction == 0) {
            uint32FloatValue.uint32Value = (sign << 31U);
        } else {
            exponent = 127U - 14U;
            while ((fraction & (1 << 10U)) == 0) {
                exponent--;
                fraction <<= 1;
            }
            fraction &= 0x3FF;
            uint32FloatValue.uint32Value = (sign << 31U) | (exponent << 23U) | (fraction << 13U);
        }
    } else if (exponent == 0x1F) {
        uint32FloatValue.uint32Value = (sign << 31U) | (0xFF << 23U) | (fraction << 13U);
    } else {
        uint32FloatValue.uint32Value = (sign << 31U) | ((exponent + (127U - 15U)) << 23U) | (fraction << 13U);
    }

    return uint32FloatValue.float32Value;
}

inline float DecodeBfp16(uint16_t float16Value)
{
    uint32_t sign = float16Value >> 15;
    uint32_t exponent = (float16Value >> 7) & 0xFF;
    uint32_t fraction = (float16Value & 0x7F);
    UInt32Float uint32FloatValue(0.0f);
    uint32FloatValue.uint32Value = (sign << 31U) | (exponent << 23U) | (fraction << 16U);
    return uint32FloatValue.float32Value;
}

}  // namespace

mindie_llm::cpu_logits_handler::PostProcessing::PostProcessing() : conf(nullptr), dictConf(nullptr),
    score16(nullptr), score32(nullptr), index(nullptr),
    scoreSize(0), result(nullptr), logprobs(nullptr), batchSize(0), maxLogprobs(0),
    dtype(mindie_llm::cpu_logits_handler::Dtype::FLOAT16), scoreSizeReal(0), speedMode(false), useApprox(false)
{}

mindie_llm::cpu_logits_handler::PostProcessing::~PostProcessing()
{}

void mindie_llm::cpu_logits_handler::PostProcessing::Init(
    std::map<int, mindie_llm::cpu_logits_handler::Configure> *dictConfIn,
    absl::Span<int> requestIdsIn, uint16_t *score16In, float *score32In, uint64_t *indexIn,
    int scoreSizeIn, int *resultIn, float *logprobsIn, int batchSizeIn, int maxLogprobsIn,
    std::string dTypeStr, bool speedModeIn, bool useApproxIn)
{
    if (scoreSizeIn < 1) {
        throw std::invalid_argument("The input score size is less than 1.");
    }
    if (requestIdsIn.size() < static_cast<std::size_t>(batchSizeIn)) {
        throw std::invalid_argument("The input requestIdsIn size is less than batchSizeIn.");
    }
    this->dictConf = dictConfIn;
    this->requestIds = requestIdsIn;
    this->score16 = score16In;
    this->score32 = score32In;
    this->index = indexIn;
    this->scoreSize = scoreSizeIn;
    this->result = resultIn;
    this->logprobs = logprobsIn;
    this->batchSize = batchSizeIn;
    this->maxLogprobs = maxLogprobsIn;
    this->dtype = mindie_llm::cpu_logits_handler::GetDtype(dTypeStr);
    this->speedMode = speedModeIn;
    this->useApprox = useApproxIn;
}

void mindie_llm::cpu_logits_handler::PostProcessing::Run()
{
    if (dictConf == nullptr) {
        throw std::invalid_argument("dictConf is null.");
    }
    std::thread::id threadId = std::this_thread::get_id();
    std::stringstream ss;
    ss << threadId;
    threadIdStr = ss.str();
    for (int i = 0; i < batchSize; i++) {
        if ((*dictConf).find(requestIds[i]) == (*dictConf).end()) {
            DecodeBySize(scoreSize);
            ArgMax(false);
        } else {
            conf = &((*dictConf).at(requestIds[i]));
            if (conf->logprobs < 0 || conf->logprobs > maxLogprobs) {
                throw std::invalid_argument("The logprobs is < 0 or > maxLogprobs.");
            }
            if (conf->sample) {
                DoTopK();
                DoTopP();
                Sampling(conf->logprobs);
            } else {
                DecodeBySize(scoreSize);
                ArgMax(false, conf->logprobs);
            }
        }
        score16 += scoreSize;
        score32 += scoreSize;
        index += scoreSize;
        result += (maxLogprobs + 1);
        logprobs += (maxLogprobs + 1);
    }
}

void mindie_llm::cpu_logits_handler::PostProcessing::DoTopK()
{
    if (conf->topK > 0 && conf->topK < scoreSize) {
        DecodeBySize(conf->topK);
    } else {
        DecodeBySize(scoreSize);
    }
}

void mindie_llm::cpu_logits_handler::PostProcessing::DoTopP()
{
    if (std::fabs(conf->topP - 1.0) > 1e-9 and useApprox) {
        TopApprox();
    } else if (std::fabs(conf->topP - 1.0) > 1e-9 and not useApprox) {
        TopP();
    }
}

void mindie_llm::cpu_logits_handler::PostProcessing::WindowSlideApprox(const float &topP,
    int &pNum, float &cumSum, int &windowSize, const int &scoreIndexSize)
{
    if (windowSize <= 0 || windowSize >= scoreIndexSize) {
        throw std::invalid_argument("The window size is less than 1 or greater than scoreIndexSize.");
    }
    std::deque<float> slideWindow;
    slideWindow.resize(windowSize);
    float windowSum = 0;
    for (int i = 0; i < windowSize; i++) {
        DecodeByDtypeElement(i);
        auto expNow = exp(scoreIndex[i].first);
        slideWindow[i] = expNow;
        windowSum += expNow;
    }

    int idx = windowSize;
    for (; idx < scoreIndexSize; idx++) {
        DecodeByDtypeElement(idx);
        auto expNow = exp(scoreIndex[idx].first);
        auto front = slideWindow.front();
        slideWindow.pop_front();
        slideWindow.push_back(expNow);
        cumSum += front;
        windowSum = windowSum - front + expNow;
        pNum++;
        if (cumSum > topP * cumSum + topP * (scoreIndexSize - idx + windowSize - 1) * windowSum / windowSize) {
            break;
        }
    }

    if (scoreIndexSize <= idx) {
        auto allSum = cumSum;
        for (int i = 0; i < windowSize; i++) {
            auto expNow = slideWindow[i];
            allSum += expNow;
        }
        for (int i = 0; i < windowSize; i++) {
            auto expNow = slideWindow[i];
            cumSum += expNow;
            pNum++;
            if (cumSum > topP * allSum) {
                break;
            }
        }
    }
}

void mindie_llm::cpu_logits_handler::PostProcessing::TopApprox()
{
    int scoreIndexSize = static_cast<int>(scoreIndex.size());
    for (int i = 0; i < scoreIndexSize; i++) {
        scoreIndex[i].second = i;
    }
    float topP = conf->topP;
    int pNum = 0;
    float cumSum = 0;
    int windowSize = 1000;
    if (scoreIndexSize > windowSize) {
        WindowSlideApprox(topP, pNum, cumSum, windowSize, scoreIndexSize);
    } else {
        SoftmaxNoModify();
        for (int i = 0; i < scoreIndexSize; i++) {
            pNum++;
            cumSum += softmaxIndex[i].first;
            if (cumSum > topP) {
                break;
            }
        }
    }
    scoreIndex.resize(pNum);
}

void mindie_llm::cpu_logits_handler::PostProcessing::TopP()
{
    SoftmaxNoModify(false);
    float topP = conf->topP;
    int pNum = 0;
    float cumSum = 0;

    if (scoreSizeReal < 0 || scoreSizeReal > MAX_SCORE_SIZE) {
        throw std::invalid_argument("scoreSizeReal must be non-negative and within the maximum allowed size");
    }

    for (int i = 0; i < scoreSizeReal; i++) {
        pNum++;
        cumSum += softmaxIndex[i].first;
        if (cumSum > topP) {
            break;
        }
    }

    scoreIndex.resize(pNum);
}

void mindie_llm::cpu_logits_handler::PostProcessing::Sampling(const int &numLogprobs)
{
    SoftmaxNoModify();

    if (conf->sampleMethod == mindie_llm::cpu_logits_handler::SamplerType::EXPONENTIAL) {
        RandomExpDistribution(numLogprobs);
        ArgMax(true);
    } else if (conf->sampleMethod == mindie_llm::cpu_logits_handler::SamplerType::MULTINOMIAL) {
        MultinomialSample(numLogprobs);
    } else {
        LOG_ERROR_LLM << "sampler type [" << static_cast<int>(conf->sampleMethod) << "]";
    }
}

void mindie_llm::cpu_logits_handler::PostProcessing::SoftmaxNoModify(bool indexSecond)
{
    double sumT = 0;
    double maxValue = -DBL_MAX;
    softmaxIndex.resize(scoreIndex.size());

    for (size_t i = 0; i < scoreIndex.size(); i++) {
        maxValue = scoreIndex[i].first > maxValue ? scoreIndex[i].first : maxValue;
    }
    for (size_t i = 0; i < scoreIndex.size(); i++) {
        softmaxIndex[i].first = exp(scoreIndex[i].first - maxValue);
        softmaxIndex[i].second = indexSecond ? static_cast<int>(scoreIndex[i].second) : static_cast<int>(i);
        sumT += softmaxIndex[i].first;
    }
    if (scoreIndex.size() == 0) {
        throw std::runtime_error("The score index is empty.");
    } else if (std::fabs(sumT - 0) < 1e-10) {
        for (size_t i = 0; i < scoreIndex.size(); i++) {
            softmaxIndex[i].first = 1.0f / scoreIndex.size();
        }
    } else {
        for (size_t i = 0; i < scoreIndex.size(); i++) {
            softmaxIndex[i].first /= sumT;
        }
    }
}

void mindie_llm::cpu_logits_handler::PostProcessing::WriteTopResult(const int &numLogprobs)
{
    if (!speedMode) {
        for (int i = 0; i < numLogprobs; ++i) {
            if (i < static_cast<int>(softmaxIndex.size())) {
                *(logprobs + i + 1) = logf(softmaxIndex[i].first);
            }
            mindie_llm::CheckParamRange(i, 0, scoreSize - 1);
            *(result + i + 1) = index[i];
        }
    } else {
        for (int i = 0; i < numLogprobs; ++i) {
            if (i < static_cast<int>(softmaxIndex.size())) {
                *(logprobs + i + 1) = logf(softmaxIndex[i].first);
            }
            *(result + i + 1) = i;
        }
    }
}

void mindie_llm::cpu_logits_handler::PostProcessing::RandomExpDistribution(const int &numLogprobs)
{
    WriteTopResult(numLogprobs);

    for (size_t i = 0; i < softmaxIndex.size(); ++i) {
        if (std::fabs(conf->randomValue - 0) < 1e-10) {
            throw std::runtime_error("The random number is zero in exponential distribution.");
        }
        softmaxIndex[i].first /= conf->randomValue;
    }
}

void mindie_llm::cpu_logits_handler::PostProcessing::ArgMax(bool sample, const int &numLogprobs)
{
    float maxValue;
    int maxIndex;

    if (sample) {
        maxValue = softmaxIndex[0].first;
        maxIndex = softmaxIndex[0].second;
        for (auto &softmax : softmaxIndex) {
            if (softmax.first > maxValue) {
                maxValue = softmax.first;
                maxIndex = softmax.second;
            }
        }
        if (numLogprobs > 0) {
            *logprobs = logf(maxValue);
        }
    } else {
        maxValue = scoreIndex[0].first;
        maxIndex = scoreIndex[0].second;
        for (auto &score : scoreIndex) {
            if (score.first > maxValue) {
                maxValue = score.first;
                maxIndex = score.second;
            }
        }
        if (numLogprobs > 0) {
            SoftmaxNoModify();
            WriteTopResult(numLogprobs);
            *logprobs = logf(softmaxIndex[maxIndex].first);
        }
    }

    if (!speedMode) {
        maxIndex = mindie_llm::CheckParamRange(maxIndex, 0, scoreSize - 1);
        *result = index[maxIndex];
    } else {
        *result = maxIndex;
    }
}

void mindie_llm::cpu_logits_handler::PostProcessing::MultinomialSample(const int &numLogprobs)
{
    float sum = 0;
    std::vector<float> softmaxTmp;
    softmaxTmp.resize(softmaxIndex.size());
    for (size_t i = 0; i < softmaxIndex.size(); i++) {
        sum += softmaxIndex[i].first;
        softmaxTmp[i] = sum;
    }
    float multinomialSampleNum1 = 1.00001f;
    float multinomialSampleNum2 = 0.99999f;
    if ((sum > 0) && !((sum < multinomialSampleNum1) && (sum > multinomialSampleNum2))) {
        for (auto &softmax : softmaxTmp) {
            softmax /= sum;
        }
    }

    int rightPointer = static_cast<int>(softmaxIndex.size()) - 1;
    int leftPointer = 0;

    while (rightPointer - leftPointer > 0) {
        int midPointer = leftPointer + (rightPointer - leftPointer) / 2;
        float cumProb = softmaxTmp[midPointer];
        if (cumProb < conf->randomValue) {
            leftPointer = midPointer + 1;
        } else {
            rightPointer = midPointer;
        }
    }
    int sampleIdx = leftPointer;

    if (sampleIdx < 0 || static_cast<size_t>(sampleIdx) >= softmaxIndex.size()) {
        throw std::invalid_argument("sampleIdx must be non-negative and within the maximum allowed size");
    }

    if (!speedMode) {
        *result = index[softmaxIndex[sampleIdx].second];
    } else {
        *result = softmaxIndex[sampleIdx].second;
    }
    float logprobsValue = logf(softmaxIndex[sampleIdx].first);
    *logprobs = logprobsValue;
    WriteTopResult(numLogprobs);
}

inline void mindie_llm::cpu_logits_handler::PostProcessing::DecodeBySize(const int &size)
{
    if (size < 0 || size > MAX_SCORE_SIZE) {
        throw std::invalid_argument("Invalid score size: " + std::to_string(size));
    }
    scoreSizeReal = size;
    scoreIndex.resize(scoreSizeReal);
    sortedLogits.resize(scoreSizeReal);
    DecodeByDtype();
}

void mindie_llm::cpu_logits_handler::PostProcessing::DecodeByDtype()
{
    switch (dtype) {
        case mindie_llm::cpu_logits_handler::Dtype::FLOAT16:
            for (int i = 0; i < scoreSizeReal; i++) {
                sortedLogits[i] = DecodeFp16(score16[i]);
            }
            break;
        case mindie_llm::cpu_logits_handler::Dtype::BFLOAT16:
            for (int i = 0; i < scoreSizeReal; i++) {
                sortedLogits[i] = DecodeBfp16(score16[i]);
            }
            break;
        case mindie_llm::cpu_logits_handler::Dtype::FLOAT32:
            for (int i = 0; i < scoreSizeReal; i++) {
                sortedLogits[i] = score32[i];
            }
            break;
        default:
            LOG_ERROR_LLM << "dtype " << static_cast<int>(dtype);
    }

    for (int i = 0; i < scoreSizeReal; i++) {
        scoreIndex[i] = std::pair<float, int>(sortedLogits[i], i);
    }
}

void mindie_llm::cpu_logits_handler::PostProcessing::DecodeByDtypeElement(int i)
{
    switch (dtype) {
        case mindie_llm::cpu_logits_handler::Dtype::FLOAT16:
            scoreIndex[i].first = DecodeFp16(score16[i]);
            break;

        case mindie_llm::cpu_logits_handler::Dtype::BFLOAT16:
            scoreIndex[i].first = DecodeBfp16(score16[i]);
            break;

        case mindie_llm::cpu_logits_handler::Dtype::FLOAT32:
            scoreIndex[i].first = score32[i];
            break;

        default:
            LOG_ERROR_LLM << "dtype " << static_cast<int>(dtype);
    }
}
