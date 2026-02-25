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

#ifndef MINDIE_LLM_COLLECTIVE_COMMUNICATION_OPERTATION_H
#define MINDIE_LLM_COLLECTIVE_COMMUNICATION_OPERTATION_H

#include <memory>

#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>

#include "basic_types.h"

namespace mindie_llm {
class ProcessGroup {
public:
    /**
     * @brief 获取ProcessGroup单例，参数仅在第一次调用时生效，后续使用可以不传递参数
     * @details 该函数是线程安全的，保证在多线程环境下只创建一个ProcessGroup实例
     * @param masterAddr 主节点地址
     * @param masterPort 主节点端口
     * @param rank 当前进程的rank
     * @param worldSize 全局进程数
     * @param isMaster 是否为主节点
     * @return ProcessGroup实例
     */
    static ProcessGroup &GetInstance(const std::string &masterAddr = "", uint16_t masterPort = 0,
                                     const std::string &localAddr = "", int rank = 0, int worldSize = 0,
                                     bool isMaster = false, int timeoutInSeconds = 120);

    /**
     * @brief 进程组间allgather通信
     * @details 要求tensor的shape必须保持一致
     * @param inputs allgather的通信内容, 要求指定device=torch::kCPU, 如：torch.tensor({ 1, 2}, torch::kCPU);
     * @return allgather通信结果，shape={inputs.size(), inputs.size() * world_size}，且输入输出tensor长度一致
     */
    std::vector<std::vector<torch::Tensor>> AllGather(std::vector<torch::Tensor> &inputs);

    /**
     * @brief 进程组间进行allReduce通信
     * @details 要求tensor的shape必须保持一致
     * @param tensor allreduce的通信内容
     * @param options allreduce的执行什么运算，如SUM
     */
    void AllReduce(std::vector<torch::Tensor> &tensor, c10d::AllreduceOptions options);

    /**
     * @brief 进程组间broadcast通信，通信结果保存在参数tensor中
     * @details 主节点向从节点进行广播，从节点收到的数据为主节点广播的inputs。要求tensor的shape必须保持一致
     * @param tensor broadcast的通信内容
     */
    void BroadCast(std::vector<torch::Tensor> &tensor);

protected:
    ProcessGroup(const std::string &masterAddr, uint16_t masterPort, const std::string &localAddr, int rank,
                 int worldSize, bool isMaster, int timeoutInSeconds = 120);

private:
    std::string masterAddr_;

    uint16_t masterPort_;

    std::string localAddr_;

    int rank_;

    int worldSize_;

    bool isMaster_;

    std::unique_ptr<c10d::ProcessGroupGloo> processGroup_;
};

std::string GetLocalHostIP(const std::vector<NodeInfo> &nodeInfos, std::vector<std::string> &hostIps);
} // namespace mindie_llm
#endif