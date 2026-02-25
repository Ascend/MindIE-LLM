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
 
#include "common_util.h"

#include <iomanip>
#include <netinet/in.h>
#include <mutex>
#include <shared_mutex>
#include <deque>
#include <csignal>
#include <future>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <mutex>
#include <regex>
#include <ctime>
#include "file_utils.h"
#include "env_util.h"
#include "common_util.h"
#include "nlohmann/json.hpp"
#include "log.h"
#include "log/logger_def.h"

using Json = nlohmann::json;

namespace mindie_llm {
constexpr mode_t PERMISSION_RW_R_NONE = 0b110'100'000;
constexpr int MAX_IPV4_LENGTH = 32;
constexpr int MAX_IPV6_LENGTH = 128;
constexpr int DP_INDEX_RANGE = 10000;
std::vector<std::string> GetHostIP(bool skipLoopback)
{
    std::vector<std::string> ips;
    ifaddrs *ifaddr;
    ifaddrs *ifa;

    if (getifaddrs(&ifaddr) == -1) {
        perror("getifaddrs error");
        return ips;
    }

    for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr) {
            continue;
        }
        int family = ifa->ifa_addr->sa_family;
        if (family != AF_INET && family != AF_INET6) {
            continue;
        }
        if (skipLoopback && (ifa->ifa_flags & IFF_LOOPBACK)) {
            continue;
        }

        char ipstr[INET6_ADDRSTRLEN];
        void *addr = nullptr;

        if (family == AF_INET) {
            // IPv4
            addr = &(reinterpret_cast<sockaddr_in *>(ifa->ifa_addr))->sin_addr;
        } else if (family == AF_INET6) {
            // IPv6
            addr = &(reinterpret_cast<sockaddr_in6 *>(ifa->ifa_addr))->sin6_addr;
        }

        if (addr != nullptr && inet_ntop(family, addr, ipstr, sizeof(ipstr)) != nullptr) {
            ips.emplace_back(ipstr);
        }
    }

    freeifaddrs(ifaddr);
    return ips;
}

size_t GetDuration(const std::chrono::steady_clock::time_point &end, const std::chrono::steady_clock::time_point &start)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

std::string GetCurTime()
{
    auto now = std::chrono::system_clock::now();
    std::time_t nowC = std::chrono::system_clock::to_time_t(now);

    struct tm timeInfo;
    std::stringstream ss;
    if (localtime_r(&nowC, &timeInfo) == nullptr) {
        throw std::runtime_error("Failed to get local time");
    }
    ss << std::put_time(&timeInfo, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

std::vector<std::string> Split(const std::string &str, char delim)
{
    std::vector<std::string> tokens{};
    // 1. check empty string
    if (str.empty()) {
        return tokens;
    }

    auto stringFindFirstNot = [str, delim](size_t pos = 0) -> size_t {
        for (size_t i = pos; i < str.size(); i++) {
            if (str[i] != delim) {
                return i;
            }
        }
        return std::string::npos;
    };

    for (size_t lastPos = stringFindFirstNot(0); lastPos != std::string::npos;) {
        size_t pos = str.find(delim, lastPos);
        if (pos == std::string::npos) {
            tokens.emplace_back(str.substr(lastPos, str.size() - lastPos));
            break;
        }
        tokens.emplace_back(str.substr(lastPos, pos - lastPos));
        lastPos = stringFindFirstNot(pos);
    }
    return tokens;
}

std::string TrimSpace(const std::string &str)
{
    auto start = std::find_if_not(str.begin(), str.end(), [](unsigned char c) {
        return std::isspace(c);
    });
    if (start == str.end()) {
        return "";
    }

    auto end = std::find_if_not(str.rbegin(), str.rend(), [](unsigned char c) {
        return std::isspace(c);
    }).base();

    return std::string(start, end);
}

std::string ToLower(std::string str)
{
    std::transform(str.begin(), str.end(), str.begin(),
        [](unsigned char c) { return std::tolower(c); });
    return str;
}

std::string ToUpper(std::string str)
{
    std::transform(str.begin(), str.end(), str.begin(),
        [](unsigned char c) { return std::toupper(c); });
    return str;
}

bool CanonicalPath(std::string &path)
{
    if (path.empty() || path.size() > 4096L) {
        return false;
    }

    /* It will allocate memory to store path */
    auto realPath = realpath(path.c_str(), nullptr);
    if (realPath == nullptr) {
        return false;
    }

    path = realPath;
    free(realPath);
    realPath = nullptr;
    return true;
}

bool GetBinaryPath(std::string &outPath)
{
    auto miesInstallPath = EnvUtil::GetInstance().Get("MINDIE_LLM_HOME_PATH");
    if (miesInstallPath.empty()) {
        std::string linkedPath = "/proc/" + std::to_string(getpid()) + "/exe";
        std::string realPath{};
        try {
            realPath.resize(PATH_MAX);
        } catch (const std::bad_alloc& e) {
            std::cout << "[%s] " << GetCurTime() << "Failed to alloc mem" << std::endl;
            return false;
        } catch (...) {
            std::cout << "[%s] " << GetCurTime() << "Failed to resize" << std::endl;
            return false;
        }
        auto size = readlink(linkedPath.c_str(), &realPath[0], realPath.size());
        if (size < 0 || size >= PATH_MAX) {
            return false;
        }

        realPath[size] = '\0';
        std::string path{ realPath };
        outPath = path.substr(0, path.find_last_of('/'));
    } else {
        outPath = miesInstallPath.append("/bin");
    }

    return true;
}

Error GetHomePath(std::string &outHomePath)
{
    auto configPath = EnvUtil::GetInstance().Get("MINDIE_LLM_HOME_PATH");
    if (configPath.empty()) {
        /* get binary path */
        std::string binaryPath{};
        if (!(GetBinaryPath(binaryPath))) {
            std::cout << "[%s] " << GetCurTime() << "Failed to get binary path " << std::endl;
            return Error(Error::Code::ERROR, "ERROR: Failed to get binary path.");
        }

        /* get real path */
        outHomePath = binaryPath.append("/../");
    } else {
        outHomePath = configPath;
    }
    if (!CanonicalPath(outHomePath)) {
        std::cout << "[%s] " << GetCurTime() << "Failed to get real path of home " << std::endl;
        return Error(Error::Code::ERROR, "ERROR: Failed to get real path of home.");
    }
    return Error(Error::Code::OK);
}

Error GetLlmPath(std::string &outHomePath)
{
    const char *llmPath = getenv("MINDIE_LLM_HOME_PATH");
    if (llmPath == nullptr) {
        std::cout << "Failed to get MINDIE_LLM_HOME_PATH in llm_manager " << std::endl;
        return Error(Error::Code::ERROR, "ERROR: Failed to get MINDIE_LLM_HOME_PATH.");
    } else {
        std::string realPath = llmPath;
        outHomePath = realPath;
    }
    if (!CanonicalPath(outHomePath)) {
        std::cout << "Failed to get real path of MINDIE_LLM_HOME_PATH in llm_manager " << std::endl;
        return Error(Error::Code::ERROR, "ERROR: Failed to get real path of MINDIE_LLM_HOME_PATH.");
    }
    return Error(Error::Code::OK);
}

bool IsNumber(const std::string &str)
{
    // string is empty or existing space at the beginning of the string
    if (str.empty() || str[0] == ' ') {
        return false;
    }
    try {
        size_t pos;
        std::stol(str, &pos);
        return pos == str.size();
    } catch (const std::invalid_argument&) {
        return false;
    } catch (const std::out_of_range&) {
        return false;
    }
}

Error GetConfigPath(std::string &outConfigPath)
{
    auto configPath = (!outConfigPath.empty()) ?
        outConfigPath : EnvUtil::GetInstance().Get("MIES_CONFIG_JSON_PATH");
    std::string errMsg;
    std::string regularPath;
    if (!configPath.empty()) {
        bool checkFlag = FileUtils::GetCheckPermissionFlag();
        if (!FileUtils::RegularFilePath(configPath, "/", errMsg, regularPath) ||
            !FileUtils::IsFileValid(regularPath, errMsg, true, FileUtils::FILE_MODE_640, checkFlag)) {
            std::cout << errMsg << std::endl;
            return Error(Error::Code::ERROR, errMsg);
        }
        outConfigPath = regularPath;
        return Error(Error::Code::OK);
    }

    std::string homePath;
    auto res = GetHomePath(homePath);
    if (res.IsOk()) {
        outConfigPath = homePath + "/conf/config.json";
        if (!FileUtils::RegularFilePath(outConfigPath, homePath, errMsg, regularPath) ||
            !FileUtils::IsFileValid(regularPath, errMsg, true, FileUtils::FILE_MODE_640, true)) {
            std::cout << errMsg << std::endl;
            return Error(Error::Code::ERROR, errMsg);
        }

        return Error(Error::Code::OK);
    } else {
        return res;
    }
}

bool CheckAndGetLogPath(const std::string &logPath, uint64_t sizeLimit, std::string &outPath,
                        const std::string &defaultPath)
{
    bool usingDefault = logPath == defaultPath;
    std::string usingDefaultNotice = !usingDefault ? " using default path instead" : "";

    if (logPath.empty()) {
        std::cout << "logPath is empty" << usingDefaultNotice << std::endl;
        return !usingDefault && CheckAndGetLogPath(defaultPath, sizeLimit, outPath, defaultPath);
    }
    std::string path = logPath;
    std::string baseDir = "/";

    if (logPath[0] != '/') {
        // starts with '/', regarded as an absolute path
        // otherwise, regarded as a relative path
        std::string homePath{};
        if (!GetHomePath(homePath).IsOk()) {
            std::cout << "failed to get home path" << std::endl;
            return false;
        }
        baseDir = homePath;
        path = homePath + "/" + logPath;
    }

    std::regex reg(".{1,4096}");
    if (!std::regex_match(path, reg)) {
        std::cout << "The logPath is too long." << usingDefaultNotice << std::endl;
        return !usingDefault && CheckAndGetLogPath(defaultPath, sizeLimit, outPath, defaultPath);
    }

    size_t lastSlash = path.size();
    lastSlash = path.rfind('/', lastSlash - 1);
    if (lastSlash == std::string::npos) {
        std::cout << "logPath is illegal,  must such as /xxx.log." << usingDefaultNotice << std::endl;
        return !usingDefault && CheckAndGetLogPath(defaultPath, sizeLimit, outPath, defaultPath);
    }
    std::string parentPath = path.substr(0, lastSlash);
    if (!FileUtils::CheckDirectoryExists(parentPath)) {
        std::cout << "The parent path of logPath is not exist or not a dir." << usingDefaultNotice << std::endl;
        return !usingDefault && CheckAndGetLogPath(defaultPath, sizeLimit, outPath, defaultPath);
    }
    std::string errMsg{};
    std::string regularPath;
    if (!FileUtils::RegularFilePath(path, baseDir, errMsg, regularPath) ||
        !FileUtils::IsFileValid(regularPath, errMsg, false, 0b110'100'000, true, sizeLimit)) {
        std::cerr << errMsg << usingDefaultNotice << std::endl;
        return !usingDefault && CheckAndGetLogPath(defaultPath, sizeLimit, outPath, defaultPath);
    }
    outPath = regularPath;
    return true;
}

bool GetWorldSizeAndServerCountFromRanktable(size_t &tp, size_t &serverCount)
{
    auto ranktablePath = EnvUtil::GetInstance().Get("RANK_TABLE_FILE");
    // check env val RANK_TABLE_FILE
    if (ranktablePath.empty()) {
        std::cout << "env val RANK_TABLE_FILE is not exist, "
                     "please set a valid ranktable file path with RANK_TABLE_FILE environment." << std::endl;
        return false;
    }
    bool checkFlag = FileUtils::GetCheckPermissionFlag();
    std::string errMsg;
    std::string regularPath;
    if (!FileUtils::RegularFilePath(ranktablePath, "/", errMsg, regularPath) ||
        !FileUtils::IsFileValid(regularPath, errMsg, true, FileUtils::FILE_MODE_640, checkFlag)) {
        std::cout << errMsg << std::endl;
        return false;
    }
    std::ifstream rankTableFile(regularPath);
    if (!rankTableFile.is_open()) {
        std::cout << "Error: Open ranktable file failed" << std::endl;
        return false;
    }
    Json ranktableJsonData;
    rankTableFile >> ranktableJsonData;
    rankTableFile.close();
    size_t globalWorldSize = 0;
    if (ranktableJsonData.find("server_list") == ranktableJsonData.end()) {
        std::cout << "ranktable file is invalid." << std::endl;
        return false;
    }

    if (ranktableJsonData.find("server_count") == ranktableJsonData.end()) {
        std::cout << "ranktable file is invalid." << std::endl;
        return false;
    }
    Json serverListJsonData = ranktableJsonData.at("server_list");
    for (Json &serverEleData : serverListJsonData) {
        if (serverEleData.find("device") == serverEleData.end()) {
            std::cout << "ranktable file is invalid." << std::endl;
            return false;
        }
        globalWorldSize += serverEleData["device"].size();
    }
    tp = globalWorldSize;
    try {
        serverCount = std::stoul(ranktableJsonData["server_count"].get<std::string>());
    } catch (const std::exception &e) {
        std::cout << "server_count in ranktale file is invalid." << std::endl;
        return false;
    } catch (...) {
        std::cout << "server_count in ranktale file is invalid." << std::endl;
        return false;
    }
    return true;
}

void GetModelInfo(const std::string &configPath, std::string &modelName, size_t &tp, size_t &serverCount)
{
    std::string configPathTmp = configPath;
    if (configPathTmp.empty()) {
        GetConfigPath(configPathTmp);
    }
    std::string errmsg;
    std::string regularPath;
    if (!FileUtils::RegularFilePath(configPathTmp, errmsg, regularPath)) {
        MINDIE_LLM_LOG_ERROR("Path validation for \"config.json\" failed." << errmsg);
        return;
    }
    bool checkFlag = FileUtils::GetCheckPermissionFlag();
    FileValidationParams params = {true, MAX_CONFIG_PERM, MAX_CONFIG_FILE_SIZE_LIMIT, checkFlag};
    if (!FileUtils::IsFileValid(configPathTmp, errmsg, params)) {
        MINDIE_LLM_LOG_ERROR("File validation for \"config.json\" failed." << errmsg);
        return;
    }
    std::ifstream file(regularPath);
    if (!file.is_open()) {
        MINDIE_LLM_LOG_ERROR("Error: Open config json file failed, the file path: " << regularPath);
        return;
    }
    Json configJsonData;
    file >> configJsonData;
    file.close();

    bool multiNodesInferEnabled = false;
    try {
        Json backendConfig = configJsonData.at("BackendConfig");
        Json modelDeployConfig = backendConfig["ModelDeployConfig"];
        modelName = modelDeployConfig.at("ModelConfig").at(0)["modelName"];
        if (backendConfig.contains("multiNodesInferEnabled")) {
            multiNodesInferEnabled = backendConfig["multiNodesInferEnabled"];
        }
        if (!multiNodesInferEnabled) {
            tp = modelDeployConfig["ModelConfig"][0]["worldSize"];
            serverCount = 1;
        }
    } catch (const Json::exception &e) {
        MINDIE_LLM_LOG_ERROR("Config json is invalid." << e.what());
        return;
    }
}

bool GetModelInfo(std::string &modelName, size_t &tp, size_t &serverCount)
{
    std::string jsonPathTmp;
    GetConfigPath(jsonPathTmp);

    char jsonRealPathTmp[PATH_MAX] = {0x00};
    if (realpath(jsonPathTmp.c_str(), jsonRealPathTmp) == nullptr) {
        return false;
    }
    std::ifstream file(jsonRealPathTmp);
    if (!file.is_open()) {
        std::cout << "Error: Open config json file failed" << std::endl;
        return false;
    }
    Json configJsonData;
    file >> configJsonData;
    file.close();

    bool multiNodesInferEnabled = false;
    try {
        Json backendConfig = configJsonData.at("BackendConfig");
        Json modelDeployConfig = backendConfig["ModelDeployConfig"];
        modelName = modelDeployConfig["ModelConfig"][0]["modelName"];
        if (backendConfig.contains("multiNodesInferEnabled")) {
            multiNodesInferEnabled = backendConfig["multiNodesInferEnabled"];
        }
        if (!multiNodesInferEnabled) {
            tp = modelDeployConfig["ModelConfig"][0]["worldSize"];
            serverCount = 1;
        }
    } catch (const Json::exception &e) {
        std::cout << "config json is invalid." << e.what() << std::endl;
        return false;
    }

    if (multiNodesInferEnabled) {
        if (!GetWorldSizeAndServerCountFromRanktable(tp, serverCount)) {
            return false;
        }
    }
    return true;
}

void ExecuteAction(std::function<void()> action, uint32_t timeoutSeconds, std::function<void()> timeoutHandler)
{
    std::promise<bool> promise;

    std::thread worker([&action, &promise]() {
        action();
        promise.set_value(true);
    });

    if (promise.get_future().wait_for(std::chrono::seconds(timeoutSeconds)) != std::future_status::ready) {
        timeoutHandler();
    } else {
        worker.join();
    }
}

std::string JoinStrings(const std::vector<std::string> &stringsVec, const std::string &delimiter)
{
    std::ostringstream oss;
    for (size_t i = 0; i < stringsVec.size(); ++i) {
        oss << stringsVec[i];
        if (i != stringsVec.size() - 1) {
            oss << delimiter;
        }
    }
    return oss.str();
}

uint32_t RandomNumber(uint32_t maxNumber)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(0, maxNumber);
    return dis(gen);
}

std::vector<std::string> SplitPath(const std::string &absPath) noexcept
{
    std::vector<std::string> components{};
    std::string component;
    for (char ch : absPath) {
        if (ch == '/') {
            if (!component.empty()) {
                components.push_back(component);
                component.clear();
            }
        } else {
            component += ch;
        }
    }
    if (!component.empty()) {
        components.push_back(component);
    }
    return components;
}

std::string AbsoluteToAnonymousPath(const std::string &absPath) noexcept
{
    if (absPath.empty() || absPath[0] != '/') {
        return "";
    }
    std::vector<std::string> components = SplitPath(absPath.substr(1));
    uint8_t dirLevel = 2;
    if (components.size() >= dirLevel) {
        components[0] = "******";
        components[1] = "******";
    } else if (components.size() == 1) {
        components[0] = "******";
    }
    std::string anonymousPath = "/";
    for (size_t i = 0; i < components.size(); ++i) {
        anonymousPath += components[i];
        if (i < components.size() - 1) {
            anonymousPath += "/";
        }
    }

    return anonymousPath;
}

std::string AbsoluteToRelativePath(const std::string &absPath, const std::string &absDir) noexcept
{
    std::string relativePath{};
    if (absPath.empty()) {
        return relativePath;
    }
    std::string absRealPath{absPath};
    if (!CanonicalPath(absRealPath)) {
        return AbsoluteToAnonymousPath(absPath);
    }
    if (absDir.empty()) {
        return AbsoluteToAnonymousPath(absRealPath);
    }
    std::string absRealDir{absDir};
    if (!CanonicalPath(absRealDir)) {
        return AbsoluteToAnonymousPath(absRealPath);
    }
    std::string::size_type position = absRealPath.rfind(absRealDir);
    if (position == std::string::npos || position != 0) {
        return AbsoluteToAnonymousPath(absRealPath);
    }
    try {
        relativePath = absRealPath.substr(position + absRealDir.length());
        return relativePath;
    } catch (const std::exception &e) {
        return AbsoluteToAnonymousPath(absRealPath);
    }
}

std::string CleanStringForJson(const std::string &input)
{
    std::string result = "";
    auto it = input.begin();
    while (it != input.end()) {
        auto ch = static_cast<unsigned char>(*it);
        // 处理多字节UTF-8字符（包括中文）
        if (ch >= 0x80) {
            do {
                result += *it++;
            } while (it != input.end() && (static_cast<unsigned char>(*it) & 0xC0) == 0x80);
        } else {
            // 保留JSON支持的打印字符和控制字符
            if ((ch >= 0x20 && ch <= 0x7E) || ch == '\n' || ch == '\r' || ch == '\t') {
                result += ch;
            }
            ++it;
        }
    }
    return result;
}

bool IsFloatEquals(float a, float b) { return std::fabs(a - b) < 1e-6; }

std::vector<std::string> SplitString(const std::string &str, char delimiter)
{
    std::vector<std::string> result;
    std::string token;
    std::istringstream stream(str);
    while (std::getline(stream, token, delimiter)) {
        result.push_back(token);
    }
    return result;
}

std::pair<uint32_t, uint32_t> ReverseDpInstId(uint64_t dpInstanceId)
{
    uint32_t pid = static_cast<uint32_t>(dpInstanceId / DP_INDEX_RANGE);       // 获取pid
    uint32_t dpIdx = static_cast<uint32_t>(dpInstanceId % DP_INDEX_RANGE);  // 获取dp_index
    return std::make_pair(pid, dpIdx);
}

bool CheckSystemConfig(const std::string &jsonPath, Json &inputJsonData, std::string paramType)
{
    std::string homePath;
    if (!GetHomePath(homePath).IsOk()) {
        std::cout << "Error: Get home path failed." << std::endl;
        return false;
    }
    std::string systemConfigPath = homePath + "/conf/config.json";
    std::string baseDir = "/";
    if (systemConfigPath.compare(jsonPath) == 0) {
        baseDir = homePath;
    }
    return ReadJsonFile(jsonPath, baseDir, inputJsonData, paramType);
}

bool ReadJsonFile(const std::string &jsonPath, std::string &baseDir, Json &inputJsonData, std::string paramType)
{
    try {
        std::string errMsg;
        std::string regularPath;
        bool checkFlag = FileUtils::GetCheckPermissionFlag();
        FileValidationParams params = {true, FileUtils::FILE_MODE_640, MAX_CONFIG_FILE_SIZE_LIMIT, checkFlag};
        if (!FileUtils::RegularFilePath(jsonPath, baseDir, errMsg, true, regularPath) ||
            !FileUtils::IsFileValid(regularPath, errMsg, params)) {
            std::cout << errMsg << std::endl;
            return false;
        }
        std::ifstream file(regularPath);
        if (!file.is_open()) {
            std::cout << "Error: Open json file failed, the file path is " << regularPath << std::endl;
            return false;
        }
        Json jsonData;
        file >> jsonData;
        file.close();
        try {
            inputJsonData = jsonData.at(paramType);
        } catch (const Json::exception &e) {
            std::cout << paramType << ": " << e.what() << std::endl;
            return false;
        }
    } catch (const std::exception &e) {
        std::cout << "Json file is invalid. Please check json format! " << std::endl;
        return false;
    }

    return true;
}

bool CheckIp(const std::string &ipAddress, const std::string &inputName, bool enableZeroIp)
{
    if (ipAddress.empty()) {
        std::cout << "Input " << inputName << " is empty" << std::endl;
        return false;
    }

    bool isIPv6 = (ipAddress.find(':') != std::string::npos);
    bool isIPv4 = (ipAddress.find('.') != std::string::npos);

    if (isIPv6) {
        return CheckIPV6(ipAddress, inputName, enableZeroIp);
    } else if (isIPv4) {
        return CheckIPV4(ipAddress, inputName, enableZeroIp);
    } else {
        std::cout << "Input " << inputName << " [" << ipAddress <<
            "] format is invalid (neither IPv4 nor IPv6)" << std::endl;
        return false;
    }
}

bool CheckIPV4(const std::string &ipAddress, const std::string &inputName, bool enableZeroIp)
{
    if (ipAddress.empty() || ipAddress.length() > MAX_IPV4_LENGTH) {
        std::cout << "Input " << inputName << " format is invalid." << std::endl;
        return false;
    }
    
    if (!enableZeroIp && ipAddress == "0.0.0.0") {
        std::cout << "Input " << inputName << " [" << ipAddress << "] is invalid" << std::endl;
        return false;
    }

    struct in_addr addr;
    if (inet_pton(AF_INET, ipAddress.c_str(), &addr) != 1) {
        std::cout << "Input " << inputName << " [" << ipAddress << "] is invalid" << std::endl;
        return false;
    }
    
    return true;
}

bool CheckIPV6(const std::string &ipAddress, const std::string &inputName, bool enableZeroIp)
{
    if (ipAddress.empty() || ipAddress.length() > MAX_IPV6_LENGTH) { // IPv6最大长度
        std::cout << "Input " << inputName << " format is invalid." << std::endl;
        return false;
    }

    std::string cleanIp = ipAddress;
    if (cleanIp.front() == '[' && cleanIp.back() == ']') {
        cleanIp = cleanIp.substr(1, cleanIp.length() - 2);
    }

    if (!enableZeroIp && cleanIp == "::") {
        std::cout << "Input " << inputName << " [" << ipAddress << "] is invalid" << std::endl;
        return false;
    }

    struct in6_addr addr;
    if (inet_pton(AF_INET6, cleanIp.c_str(), &addr) != 1) {
        std::cout << "Input " << inputName << " [" << ipAddress << "] is invalid" << std::endl;
        return false;
    }
    
    return true;
}

bool ParsePortFromIp(const std::string &ipPort, uint32_t &port)
{
    size_t colonPos = ipPort.find(';');
    if (colonPos == std::string::npos) {
        return false;
    }

    // 获取端口部分并转换为 uint32_t
    std::string portStr = ipPort.substr(colonPos + 1);
    try {
        port = std::stoul(portStr);
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: Port number contains invalid characters: " << portStr << std::endl;
    }

    return true;
}

std::set<size_t> DeserializeSet(const std::string& data)
{
    std::set<size_t> resultSet{};
    std::size_t start = 0;
    std::size_t end;
    while ((end = data.find(',', start)) != std::string::npos) {
        std::string elemStr = data.substr(start, end - start);
        try {
            size_t elemUl = static_cast<size_t>(std::stoul(elemStr));
            resultSet.insert(elemUl);
            start = end + 1;
        } catch (const std::invalid_argument& e) {
            std::cout << "Invalid argument: " << e.what() << std::endl;;
            continue;
        } catch (const std::out_of_range& e) {
            std::cout << "Convert " << elemStr << "to unsigned long failed." << e.what() << std::endl;
            continue;
        } catch (...) {
            std::cout << "An unknown exception occurred when converting " <<
                elemStr << " to unsigned long." << std::endl;
            continue;
        }
    }
    // 处理最后一个元素
    if (start < data.size()) {
        std::string elemStr = data.substr(start);
        try {
            size_t elemLastUl = static_cast<size_t>(std::stoul(elemStr));
            resultSet.insert(elemLastUl);
        } catch (const std::invalid_argument& e) {
            std::cout << "Invalid argument: " << e.what() << std::endl;
        } catch (const std::out_of_range& e) {
            std::cout << "Convert " << elemStr << "to unsigned long failed." << e.what() << std::endl;
        } catch (...) {
            std::cout << "An unknown exception occurred when converting " <<
                elemStr << " to unsigned long." << std::endl;
        }
    }
    return resultSet;
}

std::string FormatGrpcAddress(const std::string& ip, const std::string& port)
{
    if ((ip.find(':') != std::string::npos)) {
        return "[" + ip + "]:" + port;
    } else {
        return ip + ":" + port;
    }
}

// safe get value from map<vector<int64_t>>
bool SafeGetMapVectorValue(const std::map<uint64_t, std::vector<int64_t>>& map,
                           uint64_t seqId,
                           size_t index,
                           int64_t& outValue,
                           const std::string& mapName) noexcept
{
    try {
        auto it = map.find(seqId);
        if (it == map.end()) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                       mapName << " sequence id " << seqId << " not found.");
            return false;
        }
        if (index >= it->second.size()) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                       mapName << " vector index out of range: seqId=" << seqId << ", index=" << index << ".");
            return false;
        }
        outValue = it->second.at(index);
        return true;
    } catch (const std::out_of_range& e) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                   mapName << " vector index out of range: seqId=" << seqId << ", index=" << index << ".");
        return false;
    } catch (const std::exception& e) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                   mapName << " access failed: " << e.what() << ".");
        return false;
    }
}

// safe get value from map<vector<float>>
bool SafeGetMapVectorValue(const std::map<uint64_t, std::vector<float>>& map,
                           uint64_t seqId,
                           size_t index,
                           float& outValue,
                           const std::string& mapName) noexcept
{
    try {
        auto it = map.find(seqId);
        if (it == map.end()) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                       mapName << " sequence id " << seqId << " not found.");
            return false;
        }
        if (index >= it->second.size()) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                       mapName << " vector index out of range: seqId=" << seqId << ", index=" << index << ".");
            return false;
        }
        outValue = it->second.at(index);
        return true;
    } catch (const std::out_of_range& e) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                   mapName << " vector index out of range: seqId=" << seqId << ", index=" << index << ".");
        return false;
    } catch (const std::exception& e) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                   mapName << " access failed: " << e.what() << ".");
        return false;
    }
}

bool StrToInt64(int64_t &dest, const std::string &str)
{
    if (str.empty()) {
        return false;
    }
    size_t pos = 0;
    try {
        dest = std::stoll(str, &pos);
    } catch(...) {
        return false;
    }
    if (pos != str.size()) {
        return false;
    }
    return true;
}

bool StrToUint64(uint64_t &dest, const std::string &str)
{
    if (str.empty()) {
        return false;
    }
    size_t pos = 0;
    try {
        dest = std::stoull(str, &pos);
    } catch(...) {
        return false;
    }
    if (pos != str.size()) {
        return false;
    }
    return true;
}

bool StrToUint32(uint32_t &dest, const std::string &str)
{
    if (str.empty()) {
        return false;
    }
    size_t pos = 0;
    try {
        unsigned long value = std::stoul(str, &pos);
        if (pos == str.size() && value <= std::numeric_limits<uint32_t>::max()) {
            dest = static_cast<uint32_t>(value);
            return true;
        }
    } catch(...) {
        return false;
    }
    return false;
}
} // namespace mindie_llm
