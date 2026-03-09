/*
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
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <fstream>
#include "common_util.h"
#include "file_utils.h"
#include "env_util.h"
#include "check_utils.h"
#include "base_config_manager.h"
#include "system_log.h"

using Json = nlohmann::json;
using namespace nlohmann::literals;

namespace mindie_llm {
constexpr uint32_t MIN_LOCAL_DEVICE_COUNT = 1;
constexpr uint32_t MAX_LOCAL_DEVICE_COUNT = 32; // 单机上限32卡
constexpr uint32_t MIN_SERVER_COUNT = 2;        // 最少2机
constexpr uint32_t MAX_SERVER_COUNT = 60;       // 最多60机

RanktableConfigManager::RanktableConfigManager()
{
    const std::string ranktablePath = EnvUtil::GetInstance().Get("RANK_TABLE_FILE");
    if (ranktablePath.empty()) {
        initFlag = false;
        std::cout << "Ranktable file is not exist, "
                     "please set a valid ranktable file path with RANK_TABLE_FILE environment."
                  << std::endl;
        return;
    }
    bool checkFlag = true;
    const std::string isCheck = EnvUtil::GetInstance().Get("MINDIE_CHECK_INPUTFILES_PERMISSION");
    if (isCheck == "0") {
        checkFlag = false;
    }
    std::string errMsg;
    std::string regularPath;
    if (!FileUtils::RegularFilePath(ranktablePath, "/", errMsg, regularPath) ||
        !FileUtils::IsFileValid(regularPath, errMsg, true, FileUtils::FILE_MODE_640, checkFlag)) {
        std::cout << errMsg << std::endl;
        initFlag = initFlag && false;
    }
    if (initFlag) {
        ranktablePath_ = regularPath;
    }
}

bool RanktableConfigManager::ReadRanktableData(uint32_t &serverCount, Json &serverListData)
{
    try {
        std::ifstream file(ranktablePath_);
        if (!file.is_open()) {
            std::cout << "Error: Open ranktable json file failed" << std::endl;
            return false;
        }
        Json jsonData;
        file >> jsonData;
        file.close();

        try {
            serverCount = static_cast<uint32_t>(std::stoi(jsonData["server_count"].get<std::string>()));
        } catch (const std::invalid_argument &e) {
            std::cout << "Invalid server_count in ranktable file" << std::endl;
            return false;
        } catch (const std::out_of_range &e) {
            std::cout << "Parameter server_count is out of uint32_t range [0, 4294967295] in ranktable file\n";
            return false;
        }

        try {
            serverListData = jsonData.at("server_list");
        } catch (const Json::exception &e) {
            std::cout << "Parameter server_list is invalid in ranktable file" << e.what() << std::endl;
            return false;
        }
    } catch (...) {
        std::cout << "Ranktable file is invalid. Please check json format! " << std::endl;
        return false;
    }

    return true;
}

// 获取容器的IP地址
std::string RanktableConfigManager::GetContainerIPAddress()
{
    auto ip = EnvUtil::GetInstance().Get("MIES_CONTAINER_IP");
    if (ip.empty()) {
        std::cout << "The env variable MIES_CONTAINER_IP isn't exist." << std::endl;
    } else {
        return ip;
    }

    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) != 0) {
        std::cout << "Error getting hostname" << std::endl;
        return "";
    }

    struct hostent *host = gethostbyname(hostname);
    if (host == nullptr) {
        std::cout << "Error getting host information" << std::endl;
        return "";
    }

    auto **addrList = reinterpret_cast<struct in_addr **>(host->h_addr_list);
    if (addrList == nullptr || addrList[0] == nullptr) {
        LOG_ERROR_LLM << "Getting IP address failed";
        return "";
    }
    std::string containerIP(inet_ntoa(*addrList[0]));

    return containerIP;
}

// 获取宿主机的IP地址
std::string RanktableConfigManager::GetHostIPAddress()
{
    auto hostIP = EnvUtil::GetInstance().Get("HOST_IP");
    if (!hostIP.empty() && CheckIp(hostIP, "HOST_IP", true)) {
        return hostIP;
    } else {
        return "";
    }
}

bool RanktableConfigManager::InitFromJson()
{
    if (ranktablePath_.empty()) {
        initFlag = false;
        LOG_ERROR_LLM << "Ranktable file path is invalid.";
        return initFlag;
    }

    uint32_t serverCount = 0;
    Json serverListJsonData;
    if (!ReadRanktableData(serverCount, serverListJsonData)) {
        initFlag = false;
        LOG_ERROR_LLM << "Failed to parse the json data of ranktable file data.";
        return initFlag;
    }

    ranktableParam_.serverCount = serverCount;

    std::string containerIP = GetContainerIPAddress();
    std::string hostIP = GetHostIPAddress();
    uint32_t globalWorldSize = 0;
    for (Json &serverEleData : serverListJsonData) {
        struct ServerEle serverEle {};
        globalWorldSize += FillServerEle(containerIP, hostIP, serverEleData, serverEle);

        ranktableParam_.worldSize = serverEle.device.size();
        if (containerIP == serverEle.containerIp || hostIP == serverEle.serverId) {
            ranktableParam_.local = serverEle;
        }
        ranktableParam_.serverList.push_back(serverEle);
    }
    for (auto &server : ranktableParam_.serverList) {
        if (server.serverId != ranktableParam_.master.serverId ||
            (server.serverId == ranktableParam_.master.serverId &&
             server.containerIp != ranktableParam_.master.containerIp)) {
            ranktableParam_.slaves.push_back(server);
        }
    }
    ranktableParam_.globalWorldSize = globalWorldSize;
    return initFlag;
}

uint32_t RanktableConfigManager::FillServerEle(const std::string &containerIP, const std::string &hostIP,
                                               Json &serverEleData, struct ServerEle &serverEle)
{
    serverEle.serverId = GetStringParamValue(serverEleData, "server_id");
    if (serverEleData.contains("container_ip")) {
        serverEle.containerIp = GetStringParamValue(serverEleData, "container_ip");
    }

    uint32_t globalWorldSize = 0;
    for (const Json &deviceEleData : serverEleData["device"]) {
        struct DeviceEle deviceEle {};
        deviceEle.deviceId = GetStringParamValue(deviceEleData, "device_id");
        deviceEle.deviceIp = GetStringParamValue(deviceEleData, "device_ip");
        deviceEle.rankId = GetStringParamValue(deviceEleData, "rank_id");
        if (deviceEle.rankId == "0") {
            ranktableParam_.master = serverEle;
            if (containerIP == serverEle.containerIp || hostIP == serverEle.serverId) {
                ranktableParam_.isMaster = true;
            }
        }
        globalWorldSize++;
        serverEle.device.push_back(deviceEle);
    }
    return globalWorldSize;
}

bool RanktableConfigManager::CheckDeviceId(const std::string &deviceIdStr) const
{
    try {
        uint32_t deviceId = static_cast<uint32_t>(std::stoi(deviceIdStr));
        bool checkDeviceIdFlag = true;
        CHECK_CONFIG_VALIDATION(checkDeviceIdFlag,
                                ParamChecker::CheckMaxMinValue<uint32_t>(deviceId, 63U, 0U, "device_id"));
        if (!checkDeviceIdFlag) {
            LOG_ERROR_LLM << "Parameter device_id is " << deviceId << ", which is out of allow range [0, 63].";
            return false;
        }
    } catch (const std::invalid_argument &e) {
        LOG_ERROR_LLM << "Parameter device_id is invalid in ranktable file.";
        return false;
    } catch (const std::out_of_range &e) {
        LOG_ERROR_LLM << "Parameter device_id is out of uint32_t range [0, 4294967295] in ranktable file.";
        return false;
    } catch (...) {
        LOG_ERROR_LLM << "Unknown exception occurred in device_id check.";
        return false;
    }
    return true;
}

bool RanktableConfigManager::CheckDeviceIp(const std::string &deviceIpStr) const
{
    bool checkDeviceIpFlag = true;
    CHECK_CONFIG_VALIDATION(checkDeviceIpFlag, CheckIp(deviceIpStr, "device_ip", false));
    if (!checkDeviceIpFlag) {
        LOG_ERROR_LLM << "Parameter device_ip is invalid in ranktable file.";
        return false;
    }
    return true;
}

bool RanktableConfigManager::CheckRankId(const std::string &rankIdStr) const
{
    try {
        uint32_t rankId = static_cast<uint32_t>(std::stoi(rankIdStr));
        bool checkRankIdFlag = true;
        CHECK_CONFIG_VALIDATION(checkRankIdFlag, ParamChecker::CheckMaxMinValue<uint32_t>(rankId, 511U, 0U, "rankId"));
        if (!checkRankIdFlag) {
            std::cout << "Parameter rankId in ranktable is " << rankId << ", which is out of allow range [0, 511].\n";
            return false;
        }
    } catch (const std::invalid_argument &e) {
        std::cout << "Parameter rankId is invalid in ranktable file." << std::endl;
        return false;
    } catch (const std::out_of_range &e) {
        std::cout << "Parameter rankId is out of uint32_t range [0, 4294967295] in ranktable file." << std::endl;
        return false;
    } catch (...) {
        std::cout << "Unknown exception occurred in rankId check." << std::endl;
        return false;
    }
    return true;
}

bool RanktableConfigManager::CheckParam()
{
    if (ranktableParam_.serverCount < MIN_SERVER_COUNT || ranktableParam_.serverCount > MAX_SERVER_COUNT) {
        initFlag = false;
        LOG_ERROR_LLM << "Parameter server_count must be in range [" << MIN_SERVER_COUNT << ", " << MAX_SERVER_COUNT
                  << "], but got " << ranktableParam_.serverCount;
    }

    if (ranktableParam_.serverCount != ranktableParam_.serverList.size()) {
        initFlag = false;
        LOG_ERROR_LLM << "Parameter server_count is " << ranktableParam_.serverCount
                  << ", which is not equal to server_list length in ranktable file, which is "
                  << ranktableParam_.serverList.size();
    }

    auto localDeviceCount = ranktableParam_.serverList[0].device.size();
    if (localDeviceCount < MIN_LOCAL_DEVICE_COUNT || localDeviceCount > MAX_LOCAL_DEVICE_COUNT) {
        initFlag = false;
        LOG_ERROR_LLM << "The number of devices on single node must be in [" << MIN_LOCAL_DEVICE_COUNT << ", "
                  << MAX_LOCAL_DEVICE_COUNT << "], but got " << localDeviceCount;
    }

    for (const ServerEle &ele : ranktableParam_.serverList) {
        if (ele.device.size() != localDeviceCount) {
            initFlag = false;
            LOG_ERROR_LLM << "The number of devices in every server node is " << ele.device.size()
                      << " which not equal in ranktable file, which is " << localDeviceCount;
            break;
        }

        if (ele.containerIp == "") {
            initFlag = false;
            LOG_ERROR_LLM << "The containerIp in the server node is empty in ranktable file.";
            break;
        }

        // check device_id, device_ip, rankId
        for (const DeviceEle &deviceEle : ele.device) {
            if (!RanktableConfigManager::CheckDeviceId(deviceEle.deviceId) ||
                !RanktableConfigManager::CheckDeviceIp(deviceEle.deviceIp) ||
                !RanktableConfigManager::CheckRankId(deviceEle.rankId)) {
                initFlag = false;
                break;
            };
        }
    }
    return initFlag;
}

const struct RanktableParam &RanktableConfigManager::GetParam() { return ranktableParam_; }
} // namespace mindie_llm
