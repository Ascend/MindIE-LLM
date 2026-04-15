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

#include "log_level_dynamic_handler.h"

#include <pthread.h>

#include <fstream>
#include <regex>

#include "common_util.h"
#include "log.h"
#include "log_utils.h"

namespace mindie_llm {

using namespace std;

LogLevelDynamicHandler::LogLevelDynamicHandler() {
    const char *miesInstallPath = std::getenv("MINDIE_LLM_HOME_PATH");
    if (miesInstallPath != nullptr) {
        miesInstallPath_ = miesInstallPath;
    } else {
        miesInstallPath_ = "";
    }
    string homePath;
    if (!GetHomePath(homePath)) {
        cout << "Failed to get home path." << endl;
        return;
    }
    jsonPath_ = homePath + "/conf/config.json";
}

LogLevelDynamicHandler::~LogLevelDynamicHandler() { Stop(); }

LogLevelDynamicHandler &LogLevelDynamicHandler::GetInstance() {
    static LogLevelDynamicHandler instance;
    return instance;
}

void LogLevelDynamicHandler::Init(size_t intervalMs, bool needWriteToFile) {
    cout << "LogLevelDynamicHandler start" << endl;
    LogLevelDynamicHandler &instance = LogLevelDynamicHandler::GetInstance();
    if (instance.isRunning_) {
        return;
    }
    const char *logLevelEnv = std::getenv("MINDIE_LOG_LEVEL");
    if (logLevelEnv != nullptr) {
        instance.defaultLogLevel_ = logLevelEnv;
    } else {
        instance.defaultLogLevel_ = "info";
    }
    instance.intervalMs_ = intervalMs;
    instance.needWriteToFile_ = needWriteToFile;
    instance.isRunning_ = true;
    instance.logLevelDynamicThread_ = std::thread([&instance]() {
        pthread_setname_np(pthread_self(), "LogLevelThread");
        while (instance.isRunning_) {
            try {
                // 检查动态日志参数是否需要刷新
                instance.GetAndSetLogConfig();
            } catch (const std::exception &e) {
                cout << "LogLevelDynamicHandler callback exception:" << e.what() << endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(instance.intervalMs_));
        }
    });
}

void LogLevelDynamicHandler::Stop() {
    LogLevelDynamicHandler &instance = LogLevelDynamicHandler::GetInstance();
    if (!instance.isRunning_) {
        return;
    }
    instance.isRunning_ = false;
    if (instance.logLevelDynamicThread_.joinable()) {
        instance.logLevelDynamicThread_.join();
    }
}

bool LogLevelDynamicHandler::CanonicalPath(std::string &path) const {
    if (path.empty() || path.size() > 4096L) {
        return false;
    }
    auto realPath = realpath(path.c_str(), nullptr);
    if (realPath == nullptr) {
        return false;
    }
    path = realPath;
    free(realPath);
    realPath = nullptr;
    return true;
}

bool LogLevelDynamicHandler::GetBinaryPath(std::string &outPath) {
    if (miesInstallPath_.empty()) {
        std::string linkedPath = "/proc/" + std::to_string(getpid()) + "/exe";
        std::string realPath{};
        try {
            realPath.resize(PATH_MAX);
        } catch (const std::bad_alloc &e) {
            std::cout << "Failed to alloc mem" << std::endl;
            return false;
        } catch (...) {
            std::cout << "Failed to resize" << std::endl;
            return false;
        }
        auto size = readlink(linkedPath.c_str(), &realPath[0], realPath.size());
        if (size < 0 || size >= PATH_MAX) {
            return false;
        }
        realPath[size] = '\0';
        std::string path{realPath};
        outPath = path.substr(0, path.find_last_of('/'));
    } else {
        outPath = miesInstallPath_.append("/bin");
    }
    return true;
}

bool LogLevelDynamicHandler::GetHomePath(std::string &outHomePath) {
    if (miesInstallPath_.empty()) {
        /* get binary path */
        std::string binaryPath{};
        if (!(GetBinaryPath(binaryPath))) {
            std::cout << "Failed to get binary path " << std::endl;
            return false;
        }
        /* get real path */
        outHomePath = binaryPath.append("/../");
    } else {
        outHomePath = miesInstallPath_;
    }
    if (!CanonicalPath(outHomePath)) {
        std::cout << "Failed to get real path of home " << std::endl;
        return false;
    }
    return true;
}

bool LogLevelDynamicHandler::CheckLogLevelRefreshConfig(const std::string &jsonPath, nlohmann::json &inputJsonData) {
    std::string homePath;
    if (!GetHomePath(homePath)) {
        std::cout << "Error: Get home path failed." << std::endl;
        return false;
    }
    std::string systemConfigPath = homePath + "/conf/config.json";
    std::string baseDir = "/";
    if (systemConfigPath.compare(jsonPath) == 0) {
        baseDir = homePath;
    }
    return ReadLogConfig(jsonPath, baseDir, inputJsonData);
}

bool LogLevelDynamicHandler::ReadLogConfig(const std::string &jsonPath, std::string &baseDir,
                                           nlohmann::json &inputJsonData) const {
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
        std::ifstream file(jsonPath);
        if (!file.is_open()) {
            std::cout << "Error: Open json file failed, the file path is " << regularPath << std::endl;
            return false;
        }
        nlohmann::json jsonData;
        file >> jsonData;
        file.close();
        try {
            inputJsonData = jsonData.at("LogConfig");
        } catch (const nlohmann::json::exception &e) {
            std::cout << "exception: " << e.what() << std::endl;
            return false;
        }
    } catch (const std::exception &e) {
        std::cout << "Json file is invalid. Please check json format! " << std::endl;
        return false;
    }
    return true;
}

void LogLevelDynamicHandler::GetAndSetLogConfig() {
    nlohmann::json logConfigJson;
    if (!CheckLogLevelRefreshConfig(jsonPath_, logConfigJson)) {
        InsertLogConfigToFile();
        return;
    }
    // 无效参数需要自动纠正为上一次有效值
    if (!CheckAndAutoCorrectInvalidParam(logConfigJson)) {
        return;
    }
    const auto &dynamicLogLevel = logConfigJson["dynamicLogLevel"];
    if (dynamicLogLevel.is_string() && dynamicLogLevel.get<std::string>().empty()) {
        // 动态日志级别修改为空，需要恢复参数为默认值
        if (!lastLogLevel_.empty()) {
            ClearDynamicLogConfigs();
        }
        return;
    }
    bool sameDynamicLogLevel = true;
    bool sameValidHours = true;
    if (CheckDynamicLogLevelChanged(dynamicLogLevel)) {
        sameDynamicLogLevel = false;
    }
    const auto &dynamicLogLevelValidHours = logConfigJson["dynamicLogLevelValidHours"];
    if (CheckValidHoursChanged(dynamicLogLevelValidHours)) {
        sameValidHours = false;
    }
    const auto &dynamicLogLevelValidTime = logConfigJson["dynamicLogLevelValidTime"];
    currentValidTime_ = dynamicLogLevelValidTime.get<std::string>();
    UpdateDynamicLogParam(sameDynamicLogLevel, sameValidHours);
    hasSetDynamicLog_ = true;
}

bool isValidTimeFormat(const std::string &timeStr) {
    if (timeStr.empty()) {
        return true;
    }
    if (timeStr.length() != 19) {  // 格式固定长度为19（如"2025-09-28 20:42:12"）
        return false;
    }
    std::tm tm = {};
    std::istringstream iss(timeStr);
    iss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    return !iss.fail();
}

bool LogLevelDynamicHandler::IsGreaterThanNow(const std::string &other) {
    std::tm targetTm = {};
    targetTm.tm_isdst = -1;
    strptime(other.c_str(), "%Y-%m-%d %H:%M:%S", &targetTm);
    std::time_t targetTime = mktime(&targetTm);
    return (targetTime > std::time(nullptr));
}

bool LogLevelDynamicHandler::CheckAndAutoCorrectInvalidParam(const nlohmann::json &logConfigJson) {
    bool ret = true;
    if (!logConfigJson["dynamicLogLevel"].is_string()) {
        ModifyLogConfigByKey("dynamicLogLevel", lastLogLevel_, false);
        ret = false;
    }
    if (!logConfigJson["dynamicLogLevelValidHours"].is_number_integer()) {
        ModifyLogConfigByKey("dynamicLogLevelValidHours", to_string(lastValidHours_), true);
        ret = false;
    } else {
        int hours = logConfigJson["dynamicLogLevelValidHours"].get<int>();
        if (hours < 1 || hours > 7 * 24) {
            ModifyLogConfigByKey("dynamicLogLevelValidHours", to_string(lastValidHours_), true);
            ret = false;
        }
    }
    if (!logConfigJson["dynamicLogLevelValidTime"].is_string() ||
        !isValidTimeFormat(logConfigJson["dynamicLogLevelValidTime"].get<std::string>())) {
        ModifyLogConfigByKey("dynamicLogLevelValidTime", lastValidTime_, false);
        ret = false;
    } else {
        // 如果dynamicLogLevelValidTime晚于当前时间，刷新为当前时间
        if (IsGreaterThanNow(logConfigJson["dynamicLogLevelValidTime"].get<std::string>())) {
            std::time_t currentTime = std::time(nullptr);
            char currentTimeBuf[20] = {0};
            strftime(currentTimeBuf, sizeof(currentTimeBuf), "%Y-%m-%d %H:%M:%S", std::localtime(&currentTime));
            std::string currentTimeStr(currentTimeBuf);
            ModifyLogConfigByKey("dynamicLogLevelValidTime", currentTimeStr, false);
            ret = false;
        }
    }
    return ret;
}

bool LogLevelDynamicHandler::CheckDynamicLogLevelChanged(const nlohmann::json &dynamicLogLevel) {
    currentLevel_ = dynamicLogLevel.get<std::string>();
    if (currentLevel_ != lastLogLevel_) {
        return true;
    }
    return false;
}

bool LogLevelDynamicHandler::CheckValidHoursChanged(const nlohmann::json &dynamicLogLevelValidHours) {
    currentValidHours_ = dynamicLogLevelValidHours.get<int>();
    if (currentValidHours_ != lastValidHours_) {
        return true;
    }
    return false;
}

void LogLevelDynamicHandler::UpdateDynamicLogParam(const bool sameDynamicLogLevel, const bool sameValidHours) {
    auto now = std::chrono::system_clock::now();
    std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
    char timeBuffer[20] = {};
    std::strftime(timeBuffer, sizeof(timeBuffer), "%Y-%m-%d %H:%M:%S", std::localtime(&nowTime));
    if (sameDynamicLogLevel) {
        if (!sameValidHours) {
            currentValidTime_ = timeBuffer;
        }
    } else {
        // 如果第一次读取到配置文件中的dynamicLogLevelValidTime就不为空，是服务重启，不更新validTime
        if (currentValidTime_.empty() || hasSetDynamicLog_) {
            currentValidTime_ = timeBuffer;
        }
    }
    UpdateDynamicLogParamToFile();
}

bool ParseTimeString(const std::string &timeStr, std::time_t &outTime) {
    std::tm tm = {};
    std::istringstream iss(timeStr);
    iss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    if (iss.fail()) {
        return false;
    }
    outTime = std::mktime(&tm);
    return true;
}

// 判断当前时间是否在有效范围内
bool LogLevelDynamicHandler::IsCurrentTimeWithinValidRange(const std::string &validTimeStr, int validHours) const {
    std::time_t startTime;
    if (!ParseTimeString(validTimeStr, startTime)) {
        return false;
    }
    const std::time_t endTime = startTime + validHours * 3600;
    const std::time_t currentTime = std::time(nullptr);
    return (currentTime >= startTime && currentTime < endTime);
}

void LogLevelDynamicHandler::UpdateDynamicLogParamToFile() {
    if (!IsCurrentTimeWithinValidRange(currentValidTime_, currentValidHours_)) {
        ClearDynamicLogConfigs();
        return;
    }
    // 若参数都未改变且还在生效区间内，无需继续
    if (lastLogLevel_ == currentLevel_ && lastValidHours_ == currentValidHours_) {
        return;
    }
    lastLogLevel_ = currentLevel_;
    lastValidHours_ = currentValidHours_;
    lastValidTime_ = currentValidTime_;
    ModifyLogConfigByKey("dynamicLogLevel", currentLevel_, false);
    ModifyLogConfigByKey("dynamicLogLevelValidTime", currentValidTime_, false);
    for (int i = 0; i < static_cast<int>(LoggerType::MAX_LOGGER_TYPE); i++) {
        LoggerType loggerType = static_cast<LoggerType>(i);
        LogLevel logLevel = DEFAULT_LOG_LEVEL;
        LogUtils::SetMindieLogParamLevel(loggerType, logLevel, currentLevel_);
        Log::SetLogLevel(loggerType, logLevel);
    }
}

void LogLevelDynamicHandler::ClearDynamicLogConfigs() {
    // dynamicLogLevel与dynamicLogLevelValidTime清空，dynamicLogLevelValidHours保留
    ModifyLogConfigByKey("dynamicLogLevel", "", false);
    ModifyLogConfigByKey("dynamicLogLevelValidHours", "2", true);
    ModifyLogConfigByKey("dynamicLogLevelValidTime", "", false);
    lastLogLevel_ = "";
    lastValidHours_ = 2;
    lastValidTime_ = "";
    // 恢复环境变量设置值
    for (int i = 0; i < static_cast<int>(LoggerType::MAX_LOGGER_TYPE); i++) {
        LoggerType loggerType = static_cast<LoggerType>(i);
        LogLevel logLevel = DEFAULT_LOG_LEVEL;
        LogUtils::SetMindieLogParamLevel(loggerType, logLevel, defaultLogLevel_);
        Log::SetLogLevel(loggerType, logLevel);
    }
}

void LogLevelDynamicHandler::InsertLogConfigToFile() {
    // 保证仅一个进程可以写配置文件
    if (!needWriteToFile_) {
        return;
    }
    std::ifstream ifs(jsonPath_);
    if (!ifs.is_open()) {
        cerr << "Failed to open config file: " << jsonPath_ << endl;
        return;
    }
    try {
        nlohmann::ordered_json config = nlohmann::ordered_json::parse(ifs);
        config["LogConfig"] = {
            {"dynamicLogLevel", ""}, {"dynamicLogLevelValidHours", 2}, {"dynamicLogLevelValidTime", ""}};
        std::ofstream out(jsonPath_);
        out << config.dump(4);
        out.close();
        cerr << "Dynamic Log Level config undetected, insert config into the file." << endl;
    } catch (nlohmann::json::parse_error &e) {
        cerr << "Failed to parse log level dynamic config file." << endl;
    } catch (std::exception &e) {
        cerr << "Failed to write back to log level dynamic config file." << endl;
    }
}

void LogLevelDynamicHandler::ModifyLogConfigByKey(const string &key, const string &value, bool isNumber) {
    // 保证仅一个进程可以写配置文件
    if (!needWriteToFile_) {
        return;
    }
    vector<string> fileContent;
    bool isModified = false;

    ifstream ifs(jsonPath_);
    if (!ifs.is_open()) {
        cerr << "Failed to open config file: " << jsonPath_ << endl;
        return;
    }

    string line;
    regex pattern("\"" + key + "\"\\s*:\\s*[^,]+");
    while (getline(ifs, line)) {
        if (regex_search(line, pattern)) {
            string modifiedLine;
            if (isNumber) {
                // 整数不带引号
                modifiedLine = regex_replace(line, pattern, "\"" + key + "\" : " + value);
            } else {
                // 字符串保留引号
                modifiedLine = regex_replace(line, pattern, "\"" + key + "\" : \"" + value + "\"");
            }
            fileContent.push_back(modifiedLine);
            isModified = true;
        } else {
            fileContent.push_back(line);
        }
    }
    ifs.close();
    if (!isModified) {
        cerr << "Key not found in config file: " << key << endl;
        return;
    }
    ofstream ofs(jsonPath_);
    if (!ofs.is_open()) {
        cerr << "Failed to write config file: " << jsonPath_ << endl;
        return;
    }

    for (const string &l : fileContent) {
        ofs << l << endl;
    }
    ofs.close();
    return;
}

}  // namespace mindie_llm
