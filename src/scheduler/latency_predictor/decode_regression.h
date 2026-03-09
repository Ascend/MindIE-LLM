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
 
#ifndef DECODE_REGRESSION
#define DECODE_REGRESSION

#include <cstdint>
#include <vector>

namespace mindie_llm {

class DecodeRegression {
public:
    DecodeRegression();
    ~DecodeRegression() = default;
    void AddDataPoint(uint32_t tokenNum, uint32_t kvBlockNum, float execTime);
    float Predict(int tokenNum, int kvBlock);
    double Determinant3by3(const std::vector<std::vector<double>>& matrix) const;
    std::vector<std::vector<double>> Multiply(const std::vector<std::vector<double>>& matrixA,
                                         const std::vector<std::vector<double>>& matrixB) const;
    std::vector<std::vector<double>> Inverse3by3(const std::vector<std::vector<double>>& matrix) const;
    std::vector<std::vector<double>> Transpose(const std::vector<std::vector<double>>& matrix) const;
    void Train();

private:
    int windowSize_ = 400;
    int storeDataLength_ = 1000;
    std::vector<std::vector<double>> featureMatrix_;
    std::vector<std::vector<double>> attributeMatrix_;
    std::vector<std::vector<double>> coefficients_ = {{0.2181, 0, 29.961}};
    int count_;
    int index_;
};
} // namespace mindie_llm

#endif // DECODE_REGRESSION