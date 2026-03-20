/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef MODEL_MODEL_TORCH_H
#define MODEL_MODEL_TORCH_H
#include <memory>
#include <string>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#include <torch/custom_class.h>
#include <torch/script.h>
#pragma GCC diagnostic pop

#include "atb_speed/base/model.h"

#include "models/llama/model/decoder_model.h"

namespace atb_speed {
class ModelTorch : public torch::CustomClassHolder {
public:
    explicit ModelTorch(std::string modelName);
    ~ModelTorch() override;
    int64_t SetParam(const std::string &param);
    int64_t SetWeight(std::vector<torch::Tensor> atWeightTensors);
    int64_t SetWeightFormatToNZ(int64_t weightId);
    int64_t SetKVCache(std::vector<torch::Tensor> atKCacheTensors, std::vector<torch::Tensor> atVCacheTensors);
    int64_t SkipEvent(bool isSkip);
    std::vector<torch::Tensor> Execute(std::vector<torch::Tensor> atInTensors, std::string param);
    int64_t ExecuteOut(std::vector<torch::Tensor> atInTensors, std::vector<torch::Tensor> atOutTensors,
        std::string param);
    int64_t UpdateWeightsPtr(std::vector<torch::Tensor> atWeightTensors, const std::vector<int64_t> &oldWeightIds);
    c10::intrusive_ptr<ModelTorch> clone() const { return c10::make_intrusive<ModelTorch>(modelName_);}
    void ClearInternalTensors() const;
    void ResetExecutionStatus() const;

private:
    void AtTensor2Tensor(std::vector<torch::Tensor> &atTensors, std::vector<atb::Tensor> &opsTensors) const;
    int64_t ExecuteOutImpl(std::vector<atb::Tensor> &inTensors, std::vector<atb::Tensor> &outTensors,
                        const std::string &param);
    void* GetWorkSpace(const uint64_t bufferSize, const uint32_t bufferKey = 0) const;
    int64_t ClearCachedWorkspace() const;
    atb::Tensor CreateInternalTensorFromDesc(const atb::TensorDesc &tensorDesc) const;
    void RunTask(std::string taskName, std::function<int()> task) const;
private:
    std::string modelName_;
    std::shared_ptr<atb_speed::Model> model_;
    uint64_t executeCount_ = 0;
    uint64_t modelId_ = 0;
    std::shared_ptr<atb::Context> context_;
    static std::vector<torch::Tensor> atInternalTensors_;
};
}

#endif