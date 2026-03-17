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

#ifndef DYNAMIC_BATCH_RECORDER_H
#define DYNAMIC_BATCH_RECORDER_H

#include <memory>
#include <mutex>
#include <unordered_map>

namespace mindie_llm {

// Forward declarations
class LatencyPredictor;
class DecodeBatchSizeTracker;

/**
 * @brief DynamicBatchRecorder records batch statistics for dynamic batch size across DPs.
 *
 * This class uses a singleton pattern where each DP rank has its own recorder instance.
 * Data from all DPs can be collected and aggregated (e.g., take maximum) for global
 * batch size decisions in DP aggregation scenarios.
 */
class DynamicBatchRecorder {
public:
    DynamicBatchRecorder(const DynamicBatchRecorder &) = delete;
    DynamicBatchRecorder &operator=(const DynamicBatchRecorder &) = delete;

    /**
     * @brief Get the instance of DynamicBatchRecorder for a specific DP rank.
     * @param localDPRank The DP rank ID
     * @return Reference to the DynamicBatchRecorder instance
     */
    static DynamicBatchRecorder &GetInstance(size_t localDPRank);

    /**
     * @brief Set the LatencyPredictor for this DP rank.
     * @param predictor Shared pointer to LatencyPredictor
     */
    void SetLatencyPredictor(const std::shared_ptr<LatencyPredictor> &predictor);

    /**
     * @brief Get the LatencyPredictor for this DP rank.
     * @return Shared pointer to LatencyPredictor
     */
    std::shared_ptr<LatencyPredictor> GetLatencyPredictor() const;

    /**
     * @brief Set the DecodeBatchSizeTracker for this DP rank.
     * @param tracker Shared pointer to DecodeBatchSizeTracker
     */
    void SetDecodeBatchSizeTracker(const std::shared_ptr<DecodeBatchSizeTracker> &tracker);

    /**
     * @brief Get the DecodeBatchSizeTracker for this DP rank.
     * @return Shared pointer to DecodeBatchSizeTracker
     */
    std::shared_ptr<DecodeBatchSizeTracker> GetDecodeBatchSizeTracker() const;

    /**
     * @brief Get the DP rank for this recorder instance.
     * @return DP rank ID
     */
    size_t GetLocalDPRank() const;

    /**
     * @brief Set the running request count for this DP rank.
     * @param runningSize Number of running requests
     */
    void SetRunningSize(size_t runningSize);

    /**
     * @brief Get the running request count for this DP rank.
     * @return Number of running requests
     */
    size_t GetRunningSize() const;

    /**
     * @brief Aggregate latency, batch size and decode request count from all DPs.
     * @param forwardNum Window size for calculating average
     * @param maxDecodeLatency Output: Maximum decode latency across all DPs
     * @param maxBatchSize Output: Maximum batch size across all DPs
     * @param maxDecodeRequestNum Output: Maximum decode request count across all DPs
     * @return Number of valid DPs that contributed data
     */
    static size_t AggregateAllFromAllDPs(size_t forwardNum, double &maxDecodeLatency,
                                         uint64_t &maxBatchSize, size_t &maxDecodeRequestNum);

    ~DynamicBatchRecorder() = default;

private:
    explicit DynamicBatchRecorder(size_t localDPRank);

private:
    size_t localDPRank_{0};
    size_t runningSize_{0};
    std::shared_ptr<LatencyPredictor> predictor_{nullptr};
    std::shared_ptr<DecodeBatchSizeTracker> decodeBatchSizeTracker_{nullptr};

    static std::unordered_map<size_t, std::unique_ptr<DynamicBatchRecorder>> instances_;
    static std::mutex mutex_;
};

} // namespace mindie_llm

#endif // DYNAMIC_BATCH_RECORDER_H
