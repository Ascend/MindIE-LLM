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

#include "process_group.h"
#include "log.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstdlib>
#include <netinet/in.h>
#include <stdexcept>
#include <sys/socket.h>
#include <unistd.h>

#include <torch/csrc/distributed/c10d/TCPStore.hpp>

namespace mindie_llm {
namespace {
int CreateMasterListenSocket(const std::string &masterAddr, uint16_t masterPort)
{
    int listenFd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listenFd < 0) {
        MINDIE_LLM_LOG_ERROR("CreateMasterListenSocket socket failed, errno=" << errno);
        return -1;
    }

    int opt = 1;
    if (::setsockopt(listenFd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        MINDIE_LLM_LOG_ERROR("CreateMasterListenSocket setsockopt SO_REUSEADDR failed, errno=" << errno);
        ::close(listenFd);
        return -1;
    }

    sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(masterPort);
    if (::inet_pton(AF_INET, masterAddr.c_str(), &addr.sin_addr) != 1) {
        MINDIE_LLM_LOG_ERROR("CreateMasterListenSocket inet_pton failed, masterAddr=" << masterAddr <<
                                                                                       ", errno=" << errno);
        ::close(listenFd);
        return -1;
    }

    if (::bind(listenFd, static_cast<sockaddr *>(static_cast<void *>(&addr)), sizeof(addr)) < 0) {
        MINDIE_LLM_LOG_ERROR("CreateMasterListenSocket bind failed, masterAddr=" << masterAddr <<
                                                                                 ", port=" << masterPort <<
                                                                                 ", errno=" << errno);
        ::close(listenFd);
        return -1;
    }

    if (::listen(listenFd, SOMAXCONN) < 0) {
        MINDIE_LLM_LOG_ERROR("CreateMasterListenSocket listen failed, errno=" << errno);
        ::close(listenFd);
        return -1;
    }

    return listenFd;
}
} // namespace

ProcessGroup &ProcessGroup::GetInstance(const std::string &masterAddr, uint16_t masterPort,
                                        const std::string &localAddr, int rank, int worldSize, bool isMaster,
                                        int timeoutInSeconds)
{
    static ProcessGroup instance(masterAddr, masterPort, localAddr, rank, worldSize, isMaster, timeoutInSeconds);
    return instance;
}

ProcessGroup::ProcessGroup(const std::string &masterAddr, uint16_t masterPort, const std::string &localAddr, int rank,
                           int worldSize, bool isMaster, int timeoutInSeconds)
    : masterAddr_(masterAddr), masterPort_(masterPort), localAddr_(localAddr), rank_(rank), worldSize_(worldSize),
      isMaster_(isMaster)
{
    MINDIE_LLM_LOG_WARN("ProcessGroup construct, masterAddr="
                        << masterAddr << ", masterPort=" << masterPort << ", localAddr=" << localAddr
                        << ", rank=" << rank << ", worldSize=" << worldSize << ", isMaster=" << isMaster
                        << ", timeoutInSeconds=" << timeoutInSeconds);
   
    try {
            // 1. 创建TCPStore
        c10d::TCPStoreOptions tcpOptions;
        tcpOptions.port = masterPort_;
        tcpOptions.isServer = isMaster_;
        tcpOptions.useLibUV = false;

        // 通过 master_listen_fd 控制 TCPStore 的监听行为，避免在 0.0.0.0 上监听。
        int masterListenFd = -1;
        if (isMaster_) {
            masterListenFd = CreateMasterListenSocket(masterAddr_, masterPort_);
            MINDIE_LLM_LOG_INFO("ProcessGroup construct CreateMasterListenSocket success, masterListenFd=" <<
                                 masterListenFd);
            if (masterListenFd < 0) {
                MINDIE_LLM_LOG_ERROR("ProcessGroup construct CreateMasterListenSocket failed.");
                throw std::runtime_error("CreateMasterListenSocket failed");
            }
            tcpOptions.masterListenFd = masterListenFd;
        }

        auto store = c10::make_intrusive<c10d::TCPStore>(masterAddr_, tcpOptions);

        // 2. 创建ProcessGroup
        auto options = c10d::ProcessGroupGloo::Options::create();
        options->timeout = std::chrono::seconds(timeoutInSeconds);
        options->devices.emplace_back(c10d::ProcessGroupGloo::createDeviceForHostname(localAddr_));
        processGroup_ = std::make_unique<c10d::ProcessGroupGloo>(store, rank_, worldSize_, options);
    } catch (const std::exception &e) {
        MINDIE_LLM_LOG_ERROR("Failed to initialize ProcessGroup: " << e.what());
        throw;
    } catch (...) {
        MINDIE_LLM_LOG_ERROR("Unknown error occurred while initializing ProcessGroup.");
        throw;
    }
}

std::vector<std::vector<torch::Tensor>> ProcessGroup::AllGather(std::vector<torch::Tensor> &inputs)
{
    std::vector<std::vector<torch::Tensor>> outputs(inputs.size());
    for (auto &item : outputs) {
        for (size_t i = 0; i < static_cast<size_t>(worldSize_) * inputs.size(); ++i) {
            item.emplace_back(torch::empty_like(inputs[0]));
        }
    }
    processGroup_->allgather(outputs, inputs)->wait();
    return outputs;
}

void ProcessGroup::AllReduce(std::vector<torch::Tensor> &tensor, c10d::AllreduceOptions options)
{
    processGroup_->allreduce(tensor, options)->wait();
}

void ProcessGroup::BroadCast(std::vector<torch::Tensor> &tensor) { processGroup_->broadcast(tensor)->wait(); }

std::string GetLocalHostIP(const std::vector<NodeInfo> &nodeInfos, std::vector<std::string> &hostIps)
{
    for (size_t i = 0; i < nodeInfos.size(); ++i) {
        if (std::find(hostIps.begin(), hostIps.end(), nodeInfos[i].hostIp) != hostIps.end()) {
            return nodeInfos[i].hostIp;
        }
    }
    return "";
}
} // namespace mindie_llm