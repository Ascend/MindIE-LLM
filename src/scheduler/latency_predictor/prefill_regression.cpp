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
 
#include "prefill_regression.h"

#include <cmath>

#include "system_log.h"

namespace mindie_llm {
PrefillRegression::PrefillRegression() : count_(0), sumX_(0), sumY_(0), sumXX_(0), sumXY_(0) {}

void PrefillRegression::AddDataPoint(float tokenNum, float execTime)
{
    count_++;
    LOG_DEBUG_LLM << "prefill predictor: AddDataPoint count: " << count_;
    LOG_DEBUG_LLM << "prefill predictor: AddDataPoint tokenNum: " << tokenNum << " execTime: " << execTime;
    LinearRegression(tokenNum, execTime);
    LOG_DEBUG_LLM << "prefill predictor: update coefficients: " << slope_ << " * tokenNum + " << intercept_;
}

float PrefillRegression::Predict(int tokenNum) const
{
    LOG_DEBUG_LLM << "prefill predictor: predict tokenNum: " << tokenNum;
    return slope_ * tokenNum + intercept_;
}

void PrefillRegression::LinearRegression(float tokenNum, float execTime)
{
    if (count_ == 1) {
        // 如果仅有一个点加入prefill拟合
        sumX_ += tokenNum;
        sumY_ += execTime;
        sumXX_ = tokenNum * tokenNum;
        sumXY_ = tokenNum * execTime;
        return;
    }
    // 非第一个点加入prefill拟合，临时参数保存拟合数值
    float tmpSumX = sumX_ + tokenNum;
    float tmpSumY = sumY_ + execTime;
    float tmpSumXX = sumXX_ + tokenNum * tokenNum;
    float tmpSumXY = sumXY_ + tokenNum * execTime;
    float denominator = count_ * tmpSumXX - tmpSumX * tmpSumX;

    if (std::fabs(denominator) > 1e-9) {
        // 分母不为0下开始拟合
        float tmpSlope = (count_ * tmpSumXY - tmpSumX * tmpSumY) / denominator;
        float tmpIntercept = (tmpSumY * tmpSumXX - tmpSumX * tmpSumXY) / denominator;
        if (tmpSlope > 0 && tmpIntercept > 0) {
            // 计算的出更新的斜率与截距均大于0，更新参数及相关计算数值
            slope_ = tmpSlope;
            intercept_ = tmpIntercept;
            sumX_ = tmpSumX;
            sumY_ = tmpSumY;
            sumXX_ = tmpSumXX;
            sumXY_ = tmpSumXY;
        } else {
            // 若参数中存在负数，不更新参数及相关计算数值
            int leastPointsCountForReg = 2;
            if (count_ == leastPointsCountForReg) {
                // 当拟合只有两点，且拟合出负数参数，重置拟合，丢弃两点
                count_ = 0;
                sumX_ = 0;
                sumY_ = 0;
                sumXX_ = 0;
                sumXY_ = 0;
            } else {
                // 当拟合超过点时，不改变原参数及相关参数，抛弃目前该点
                count_ -= 1;
            }
        }
    }
}
} // namespace mindie_llm