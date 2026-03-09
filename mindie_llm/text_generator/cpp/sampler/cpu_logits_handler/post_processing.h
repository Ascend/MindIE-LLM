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
 
#ifndef MINDIE_LLM_POST_PROCESSING_H
#define MINDIE_LLM_POST_PROCESSING_H

#include <random>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <map>
#include <chrono>
#include <absl/types/span.h>
#include "post_processing_profiler/profiler.h"
#include "system_log.h"

namespace mindie_llm {
namespace cpu_logits_handler {

enum class Dtype {
    FLOAT16,
    BFLOAT16,
    FLOAT32,
};

enum class SamplerType {
    EXPONENTIAL,
    MULTINOMIAL,
};

inline SamplerType GetSamplerType(std::string inSamplerType)
{
    if (inSamplerType == "exponential") {
        return SamplerType::EXPONENTIAL;
    } else if (inSamplerType == "multinomial") {
        return SamplerType::MULTINOMIAL;
    } else {
        LOG_WARN_LLM << "sampler type {" << inSamplerType << "}, use \"exponential\" for default";
        return SamplerType::EXPONENTIAL;
    }
}

inline Dtype GetDtype(std::string inDtype)
{
    if (inDtype == "float16" || inDtype == "fp16") {
        return Dtype::FLOAT16;
    } else if (inDtype == "bfloat16" || inDtype == "bf16") {
        return Dtype::BFLOAT16;
    } else if (inDtype == "float32" || inDtype == "fp32") {
        return Dtype::FLOAT32;
    } else {
        LOG_WARN_LLM << "dtype {" << inDtype << "}, use \"fp16\" for default";
        return Dtype::FLOAT16;
    }
}

struct Configure {
    Configure(int topK, float topP, bool sample, int logprobs, unsigned long long seed, std::string sampleMethod)
        : topK(topK), topP(topP), sample(sample), logprobs(logprobs), seed(seed),
        sampleMethod(GetSamplerType(sampleMethod))
    {
        this->g.seed(seed);
    }

    int topK = 0;
    float topP = 1;
    bool sample = false;
    int logprobs = 0;
    unsigned long long seed;
    std::mt19937 g;
    SamplerType sampleMethod = SamplerType::EXPONENTIAL;

    std::uniform_real_distribution<float> uniform{0, 1};
    std::exponential_distribution<float> d{1.0};

    float randomValue = 1.0;

    std::string GetConfig() const
    {
        std::stringstream sstream;
        sstream << "Host Sampling Configure is: ";
        sstream << "top k [" << topK << "], ";
        sstream << "top p [" << topP << "], ";
        sstream << "sample [" << sample << "], ";
        sstream << "logprobs [" << logprobs << "], ";
        sstream << "seed [" << seed << "], ";
        sstream << "sample method [" << static_cast<int>(sampleMethod) << "]";
        return sstream.str();
    }

    void UpdateRandomValue()
    {
        if (sampleMethod == SamplerType::EXPONENTIAL) {
            randomValue = d(g);
        } else {
            randomValue = uniform(g);
        }
    }
};

class PostProcessing {
public:
    PostProcessing();
    ~PostProcessing();

    void Init(std::map<int, Configure> *dictConfIn, absl::Span<int> requestIdsIn, uint16_t *score16In, float *score32In,
            uint64_t *indexIn, int scoreSizeIn, int *resultIn, float *logprobsIn, int batchSizeIn, int maxLogprobsIn,
            std::string dTypeStr, bool speedModeIn, bool useApproxIn);
    void Run();

public:
    Configure *conf;
    std::map<int, Configure> *dictConf;
    absl::Span<int> requestIds{};
    uint16_t *score16;
    float *score32;
    uint64_t *index;
    int scoreSize;
    int *result;
    float *logprobs;
    int batchSize;
    int maxLogprobs;
    Dtype dtype = Dtype::FLOAT16;

private:
    void DoTopK();
    void DoTopP();
    void TopP();
    void Sampling(const int &numLogprobs = 0);
    void WindowSlideApprox(const float &topP, int &pNum, float &cumSum, int &windowSize, const int &scoreIndexSize);
    void TopApprox();
    void ArgMax(bool sample = false, const int &numLogprobs = 0);
    void SoftmaxNoModify(bool indexSecond = true);
    void WriteTopResult(const int &numLogprobs = 0);
    void RandomExpDistribution(const int &numLogprobs = 0);
    void MultinomialSample(const int &numLogprobs = 0);
    inline void DecodeBySize(const int &size);
    void DecodeByDtype();
    void DecodeByDtypeElement(int i);

private:
    int scoreSizeReal;
    std::vector<std::pair<float, int>> scoreIndex;
    std::vector<std::pair<float, int>> softmaxIndex;
    std::vector<float> sortedLogits;
    std::string threadIdStr;
    bool speedMode;
    bool useApprox;
};
} // namespace cpu_logits_handler
} // namespace mindie_llm

#endif
