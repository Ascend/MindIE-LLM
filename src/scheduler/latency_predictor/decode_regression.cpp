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
 
#include "decode_regression.h"

#include <cmath>

#include "system_log.h"

namespace mindie_llm {
DecodeRegression::DecodeRegression() : count_(0), index_(0) {}

void DecodeRegression::AddDataPoint(uint32_t tokenNum, uint32_t kvBlockNum, float execTime)
{
    if (count_ % windowSize_ == 0 && count_ != 0) {
        Train();
        double coefficientTokenNum = coefficients_[0][0];
        double coefficientKVBlock = coefficients_[0][1];
        double intercept = coefficients_[0][2];
        LOG_DEBUG_LLM << "decode predictor: update coefficients: " << coefficientTokenNum << " * tokenNum + "
                                                                       << coefficientKVBlock << " * kvBlockNum + "
                                                                       << intercept;
    }
    LOG_DEBUG_LLM << "decode predictor: add data: " << tokenNum << " tokenNum + " << kvBlockNum << " kvBlockNum + "
                                                        << execTime << " time";
    index_ = count_ % storeDataLength_;
    std::vector<double> data = {static_cast<double>(tokenNum), static_cast<double>(kvBlockNum), static_cast<double>(1)};
    std::vector<double> dataAttribute = {static_cast<double>(execTime)};
    if (count_ >= storeDataLength_) {
        featureMatrix_[index_] = data;
        attributeMatrix_[index_] = dataAttribute;
    } else {
        featureMatrix_.push_back(data);
        attributeMatrix_.push_back(dataAttribute);
    }
    count_++;
}

float DecodeRegression::Predict(int tokenNum, int kvBlock)
{
    std::vector<std::vector<double>> datapointTest = {
        {static_cast<double>(tokenNum), static_cast<double>(kvBlock), static_cast<double>(1)}};
    std::vector<std::vector<double>> result = Multiply(coefficients_, Transpose(datapointTest));
    return static_cast<float>(result[0][0]);
}

void DecodeRegression::Train()
{
    double det = Determinant3by3(Multiply(Transpose(featureMatrix_), featureMatrix_));
    if (std::fabs(det) > 1e-9) {
        std::vector<std::vector<double>> tmp = Multiply(
            Multiply(Inverse3by3(Multiply(Transpose(featureMatrix_), featureMatrix_)), Transpose(featureMatrix_)),
            attributeMatrix_);
        double tmpCoefficientTokenNum = Transpose(tmp)[0][0];
        double tmpCoefficientKVBlock = Transpose(tmp)[0][1];
        double tmpIntercept = Transpose(tmp)[0][2];
        if (tmpCoefficientTokenNum > 0 && tmpCoefficientKVBlock > 0 && tmpIntercept > 0) {
            LOG_DEBUG_LLM << "decode predictor: Update happened and all parameters are positive.";
            coefficients_ = Transpose(tmp);
        } else {
            LOG_DEBUG_LLM << "decode predictor: Update failed and will still use parameters in last round.";
        }
    }
}

double DecodeRegression::Determinant3by3(const std::vector<std::vector<double>> &matrix) const
{
    // 计算3x3矩阵行列式
    double det = 0.0;
    int endDim = 2;
    det = matrix[0][0] * (matrix[1][1] * matrix[endDim][endDim] - matrix[1][endDim] * matrix[endDim][1]) -
          matrix[0][1] * (matrix[1][0] * matrix[endDim][endDim] - matrix[1][endDim] * matrix[endDim][0]) +
          matrix[0][endDim] * (matrix[1][0] * matrix[endDim][1] - matrix[1][1] * matrix[endDim][0]);
    return det;
}

std::vector<std::vector<double>> DecodeRegression::Multiply(const std::vector<std::vector<double>> &matrixA,
                                                            const std::vector<std::vector<double>> &matrixB) const
{
    size_t rowsA = matrixA.size();
    size_t colsA = matrixA[0].size();
    size_t colsB = matrixB[0].size();
    std::vector<std::vector<double>> result(rowsA, std::vector<double>(colsB, 0.0));

    for (size_t i = 0; i < rowsA; ++i) {
        for (size_t j = 0; j < colsB; ++j) {
            for (size_t k = 0; k < colsA; ++k) {
                result[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
    return result;
}

std::vector<std::vector<double>> DecodeRegression::Inverse3by3(const std::vector<std::vector<double>> &matrix) const
{
    double det = Determinant3by3(matrix);
    int inverseDim = 3;
    double inverseInit = 0.0;
    std::vector<std::vector<double>> inv(inverseDim, std::vector<double>(inverseDim, inverseInit));
    if (std::fabs(det) > 1e-9) {
        // 3X3逆矩阵求解
        int endDim = 2;
        inv[0][0] = (matrix[1][1] * matrix[endDim][endDim] - matrix[1][endDim] * matrix[endDim][1]) / det;
        inv[0][1] = (matrix[endDim][0] * matrix[1][endDim] - matrix[1][0] * matrix[endDim][endDim]) / det;
        inv[0][endDim] = (matrix[1][0] * matrix[endDim][1] - matrix[endDim][0] * matrix[1][1]) / det;
        inv[1][0] = (matrix[endDim][1] * matrix[0][endDim] - matrix[0][1] * matrix[endDim][endDim]) / det;
        inv[1][1] = (matrix[0][0] * matrix[endDim][endDim] - matrix[0][endDim] * matrix[endDim][0]) / det;
        inv[1][endDim] = (matrix[0][1] * matrix[endDim][0] - matrix[0][0] * matrix[endDim][1]) / det;
        inv[endDim][0] = (matrix[0][1] * matrix[1][endDim] - matrix[0][endDim] * matrix[1][1]) / det;
        inv[endDim][1] = (matrix[0][endDim] * matrix[1][0] - matrix[0][0] * matrix[1][endDim]) / det;
        inv[endDim][endDim] = (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) / det;
        return inv;
    }
    return {}; // 返回空表示错误
}

std::vector<std::vector<double>> DecodeRegression::Transpose(const std::vector<std::vector<double>> &matrix) const
{
    uint64_t rows = matrix.size();
    uint64_t cols = matrix[0].size();
    std::vector<std::vector<double>> transposed(cols, std::vector<double>(rows));
    for (uint64_t i = 0; i < rows; ++i) {
        for (uint64_t j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }
    return transposed;
}
} // namespace mindie_llm