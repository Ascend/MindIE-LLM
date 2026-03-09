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
#include "file_utils.h"

#include <iostream>
#include <sstream>
#include <cstring>
#include <unistd.h>
#include <climits>
#include <cerrno>
#include <regex>
#include <experimental/filesystem>
#include <sys/stat.h>
namespace fs = std::experimental::filesystem;
namespace {
constexpr long MIN_MALLOC_SIZE = 1;
constexpr uint64_t DEFAULT_MAX_DATA_SIZE = 4294967296;
constexpr int PER_PERMISSION_MASK_RWX = 0b111;
constexpr size_t MAX_ENV_LENGTH = 256;
} // namespace

namespace mindie_llm {

static uint64_t g_defaultMaxDataSize = DEFAULT_MAX_DATA_SIZE;
static const mode_t FILE_MODE = 0740;

static bool GetFileSize(const std::string &filePath, size_t &fileSize)
{
    if (!FileUtils::CheckFileExists(filePath)) {
        std::cerr << "File does not exist!" << std::endl;
        return false;
    }
    std::string baseDir = "/";
    std::string errMsg{};
    std::string regularPath;
    if (!FileUtils::RegularFilePath(filePath, baseDir, errMsg, regularPath)) {
        std::cerr << "Regular file failed by " << errMsg << std::endl;
        return false;
    }

    FILE *fp = fopen(regularPath.c_str(), "rb");
    if (fp == nullptr) {
        std::cerr << "File failed to open file." << std::endl;
        return false;
    }
    auto ret = fseek(fp, 0, SEEK_END);
    if (ret != 0) {
        std::cerr << "Error seeking to end of file" << std::endl;
        fclose(fp);
        return false;
    }
    fileSize = static_cast<size_t>(ftell(fp));
    ret = fseek(fp, 0, SEEK_SET);
    if (ret != 0) {
        std::cerr << "Error seeking to set of file" << std::endl;
        fclose(fp);
        return false;
    }
    fclose(fp);
    return true;
}

static std::string GetBaseFileName(const std::string &path)
{
    std::string tempPath = path;
    if (!tempPath.empty() && (tempPath.back() == '/' || tempPath.back() == '\\')) {
        tempPath.pop_back();
    }
    size_t lastSlashPos = tempPath.find_last_of("/\\");
    if (lastSlashPos == std::string::npos) {
        return tempPath;
    }
    return tempPath.substr(lastSlashPos + 1);
}

static size_t GetFileSize(const std::string &filePath)
{
    if (!FileUtils::CheckFileExists(filePath)) {
        std::cerr << "File does not exist!" << std::endl;
        return 0;
    }
    std::string baseDir = "/";
    std::string errMsg;
    std::string regularPath;
    if (!FileUtils::RegularFilePath(filePath, baseDir, errMsg, true, regularPath)) {
        std::cerr << "Regular file failed by " << errMsg << std::endl;
        return 0;
    }

    FILE *fp = fopen(regularPath.c_str(), "rb");
    if (fp == nullptr) {
        std::cerr << "Failed to open file." << std::endl;
        return 0;
    }
    int res = fseek(fp, 0, SEEK_END);
    if (res != 0) {
        std::cerr << "Failed to fseek SEEK_END." << std::endl;
        if (fclose(fp) != 0) {
            std::cerr << "File close failed." << std::endl;
        }
        return 0;
    }
    size_t fileSize = static_cast<size_t>(ftell(fp));
    res = fseek(fp, 0, SEEK_SET);
    if (res != 0) {
        std::cerr << "Failed to fseek SEEK_SET." << std::endl;
        if (fclose(fp) != 0) {
            std::cerr << "File close failed." << std::endl;
        }
        return 0;
    }
    res = fclose(fp);
    if (res != 0) {
        std::cerr << "File close failed." << std::endl;
        return 0;
    }
    return fileSize;
}

static bool CheckDataSize(uint64_t size, uint64_t maxFileSize = DEFAULT_MAX_DATA_SIZE)
{
    if (maxFileSize <= MIN_MALLOC_SIZE || maxFileSize > g_defaultMaxDataSize) {
        return false;
    }
    if ((size > maxFileSize) || (size < MIN_MALLOC_SIZE)) {
        std::cerr << "Input data size(" << size << ") out of range[" << MIN_MALLOC_SIZE << "," << maxFileSize << "]."
                  << std::endl;
        return false;
    }

    return true;
}

bool FileUtils::CheckFileExists(const std::string &filePath)
{
    struct stat buffer;
    return (stat(filePath.c_str(), &buffer) == 0);
}

bool FileUtils::CheckDirectoryExists(const std::string &dirPath)
{
    struct stat buffer;
    if (stat(dirPath.c_str(), &buffer) != 0) {
        return false;
    }
    return (S_ISDIR(buffer.st_mode) == 1);
}

bool FileUtils::IsSymlink(const std::string &filePath)
{
    struct stat buf;
    std::string normalizedPath = filePath;
    while (!normalizedPath.empty() && normalizedPath.back() == '/') {
        normalizedPath.pop_back();
    }
    if (lstat(normalizedPath.c_str(), &buf) != 0) {
        return false;
    }
    return S_ISLNK(buf.st_mode);
}

bool FileUtils::RegularFilePath(const std::string &filePath, const std::string &baseDir, std::string &errMsg, bool flag, std::string &regularPath)
{
    if (filePath.empty()) {
        errMsg = "The file path: " + filePath + " is empty.";
        return false;
    }
    if (baseDir.empty()) {
        errMsg = "The file path: " + filePath + " basedir is empty.";
        return false;
    }
    if (filePath.size() >= PATH_MAX) {
        errMsg = "The file path " + filePath + " exceeds the maximum value set by PATH_MAX.";
        return false;
    }
    if (flag) {
        if (IsSymlink(filePath)) {
            errMsg = "The file " + filePath + " is a link.";
            return false;
        }
    }

    char path[PATH_MAX] = {0x00};
    if (realpath(filePath.c_str(), path) == nullptr) {
        errMsg = "The path " + filePath + " realpath parsing failed.";
        if (errno == EACCES) {
            errMsg += " Make sure the path's owner has execute permission.";
        }
        return false;
    }
    std::string realFilePath(path, path + strnlen(path, PATH_MAX));
    if (flag) {
        std::string dir = baseDir.back() == '/' ? baseDir : baseDir + "/";
        if (realFilePath.rfind(dir, 0) != 0) {
            errMsg = "The file " + filePath + " is invalid, it's not in baseDir " + baseDir + " directory.";
            return false;
        }
    }
    regularPath = realFilePath;
    return true;
}
bool FileUtils::RegularFilePath(const std::string &filePath, const std::string &baseDir, std::string &errMsg, std::string &regularPath)
{
    if (filePath.empty()) {
        errMsg = "The file path: " + GetBaseFileName(filePath) + " is empty.";
        return false;
    }
    if (baseDir.empty()) {
        errMsg = "The file path basedir: " + GetBaseFileName(baseDir) + " is empty.";
        return false;
    }
    if (filePath.size() > PATH_MAX) {
        errMsg = "The file path: " + GetBaseFileName(filePath) + " exceeds the maximum value set by PATH_MAX.";
        return false;
    }
    if (IsSymlink(filePath)) {
        errMsg = "The file: " + GetBaseFileName(filePath) + " is a link.";
        return false;
    }
    if (CheckFileExists(filePath) || CheckDirectoryExists(filePath)) {
        char path[PATH_MAX + 1] = {0x00};
        char *ret = realpath(filePath.c_str(), path);
        if (ret == nullptr) {
            errMsg = "The path: " + GetBaseFileName(filePath) + " realpath parsing failed.";
            return false;
        }
        std::string realFilePath(path, path + strlen(path));

        std::string dir = baseDir.back() == '/' ? baseDir : baseDir + "/";
        if (realFilePath.rfind(dir, 0) != 0) {
            errMsg = "The file: " + GetBaseFileName(filePath) + " is invalid, it's not in baseDir directory.";
            return false;
        }
        regularPath = realFilePath;
    }
    regularPath = filePath;
    return true;
}
bool FileUtils::RegularFilePath(const std::string &filePath, std::string &errMsg, std::string &regularPath)
{
    if (filePath.empty()) {
        errMsg = "The file path: " + filePath + " is empty.";
        return false;
    }

    if (filePath.size() > PATH_MAX) {
        errMsg = "The file path " + filePath + " exceeds the maximum value set by PATH_MAX.";
        return false;
    }
    if (IsSymlink(filePath)) {
        errMsg = "The file " + filePath + " is a link.";
        return false;
    }
    char path[PATH_MAX] = {0x00};
    if (realpath(filePath.c_str(), path) == nullptr) {
        errMsg = "The path " + filePath + " realpath parsing failed.";
        if (errno == EACCES) {
            errMsg += " Make sure the path's owner has execute permission.";
        }
        return false;
    }
    std::string realFilePath(path, path + strlen(path));
    regularPath = realFilePath;
    return true;
}

bool FileUtils::IsFileValid(const std::string &filePath, std::string &errMsg, const FileValidationParams &params)
{
    if (!CheckFileExists(filePath)) {
        errMsg = "The input file is not a regular file or not exists";
        return !params.isFileExist;
    }
    if (!CheckDirectoryExists(filePath)) {
        size_t fileSize = GetFileSize(filePath);
        if (fileSize == 0) {
            errMsg = "The input file is empty";
        } else if (!CheckDataSize(fileSize, params.maxfileSize)) {
            errMsg = "Read input file failed, file is too large";
            return false;
        }
    }
    if (!ConstrainOwner(filePath, errMsg) || !ConstrainPermission(filePath, params.mode, errMsg)) {
        if (!params.checkPermission) {
            return true;
        }
        return false;
    }
    return true;
}

bool FileUtils::IsFileValid(const std::string &filePath, std::string &errMsg, bool isFileExit, mode_t mode,
                            bool checkPermission, uint64_t maxfileSize)
{
    if (!CheckFileExists(filePath)) {
        errMsg = "The input file: " + GetBaseFileName(filePath) + " is not a regular file or not exists";
        return !isFileExit;
    }

    if (!CheckDirectoryExists(filePath)) {
        size_t fileSize{0};
        if (!GetFileSize(filePath, fileSize)) {
            errMsg = "The input file: " + GetBaseFileName(filePath) + " failed to get file size.";
            return false;
        }
        if (!CheckDataSize(fileSize, maxfileSize)) {
            errMsg = "Read input file: " + GetBaseFileName(filePath) + " failed, file is too large";
            return false;
        }
    }

    if (!ConstrainOwner(filePath, errMsg) || !ConstrainPermission(filePath, mode, errMsg)) {
        errMsg = "Check path: " + GetBaseFileName(filePath) + " failed, by:" + errMsg;
        if (!checkPermission) {
            std::cerr << "[ERROR] " << errMsg << "; Please set: chmod 750 " << GetBaseFileName(filePath) << std::endl;
            return false;
        }
        return false;
    }

    return true;
}

bool FileUtils::IsFileAndDirectoryExists(const std::string &filePath, std::string &errMsg,
                                         const FileValidationParams &params)
{
    if (!CheckFileExists(filePath)) {
        errMsg = "The input file is not a regular file or not exists";
        return !params.isFileExist;
    }
    if (!CheckDirectoryExists(filePath)) {
        size_t fileSize = GetFileSize(filePath);
        if (fileSize == 0) {
            errMsg = "The input file is empty";
        } else if (!CheckDataSize(fileSize, params.maxfileSize)) {
            errMsg = "Read input file failed, file is too large";
            return false;
        }
    }
    return true;
}

bool FileUtils::ConstrainOwner(const std::string &filePath, std::string &errMsg)
{
    struct stat buf;
    int ret = stat(filePath.c_str(), &buf);
    if (ret != 0) {
        errMsg = "Get file stat failed.";
        return false;
    }
    if (buf.st_uid != getuid()) {
        errMsg = "owner id diff: current process user id is " + std::to_string(getuid()) + ", file owner id is " +
                 std::to_string(buf.st_uid);
        return false;
    }
    return true;
}

std::string GetModeString(const mode_t mode)
{
    std::ostringstream oss;
    oss << std::oct << (mode & (S_IRWXU | S_IRWXG | S_IRWXO));
    return oss.str();
}

bool FileUtils::ConstrainPermission(const std::string &filePath, const mode_t &mode, std::string &errMsg)
{
    struct stat buf;
    int ret = stat(filePath.c_str(), &buf);
    if (ret != 0) {
        errMsg = "Get file stat failed.";
        return false;
    }

    mode_t mask = PER_PERMISSION_MASK_RWX;
    const int perPermWidth = 3;
    std::vector<std::string> permMsg = {"Other group permission", "Owner group permission", "Owner permission"};
    for (int i = perPermWidth; i > 0; i--) {
        uint32_t curPerm = (buf.st_mode & (mask << ((i - 1) * perPermWidth))) >> ((i - 1) * perPermWidth);
        uint32_t maxPerm = (mode & (mask << ((i - 1) * perPermWidth))) >> ((i - 1) * perPermWidth);
        if ((curPerm | maxPerm) != maxPerm) {
            errMsg = " Check " + permMsg[i - 1] + " failed: " + filePath + " current permission is " +
                     std::to_string(curPerm) + ", but required no greater than " + std::to_string(maxPerm) +
                     ". Required permisssion is " + GetModeString(mode) + ", but got permission is " +
                     GetModeString(buf.st_mode);
            return false;
        }
    }
    return true;
}

bool FileUtils::GetRealFilePath(const std::string &filePath, std::string &realFilePath, std::string &errMsg) noexcept
{
    if (filePath.empty()) {
        errMsg = "[FileUtils::GetRealFilePath] filePath is empty";
        return false;
    }
    std::string workDir;
    if (!GetInstallPath(workDir, errMsg)) {
        errMsg = "[FileUtils::GetRealFilePath] get install path failed because " + errMsg;
        return false;
    }
    // 判断绝对路径和相对路径
    if (filePath.rfind(workDir) != 0) {
        realFilePath = workDir + filePath;
    } else {
        realFilePath = filePath;
    }
    std::string regularPath;
    if (!FileUtils::RegularFilePath(realFilePath, "/", errMsg, regularPath)) {
        errMsg = "[FileUtils::GetRealFilePath] file is invalid because " + errMsg;
        return false;
    }
    return true;
}

bool FileUtils::GetInstallPath(std::string &installPath, std::string &errMsg) noexcept
{
    std::string linkedPath = "/proc/" + std::to_string(getpid()) + "/exe";
    std::string realPath;
    try {
        realPath.resize(PATH_MAX);
    } catch (const std::bad_alloc &e) {
        errMsg = "Failed to alloc mem.";
        return false;
    } catch (...) {
        errMsg = "Failed to resize.";
        return false;
    }
    auto size = readlink(linkedPath.c_str(), &realPath[0], realPath.size());
    if (size < 0 || size >= PATH_MAX) {
        return false;
    }

    realPath[size] = '\0';
    std::string path{realPath};

    std::string::size_type position = path.find_last_of('/');
    if (position == std::string::npos) {
        errMsg = "get lib path failed : invalid folder path.";
        return false;
    }
    // get bin dir
    path = path.substr(0, position);

    position = path.find_last_of('/');
    if (position == std::string::npos) {
        errMsg = "get lib path failed : invalid folder path.";
        return false;
    }
    // get install dir
    path = path.substr(0, position);
    path.append("/");

    installPath = path;
    return true;
}

std::string FileUtils::GetSafeRelativePath(const std::string &oldPath)
{
    if (oldPath.empty()) {
        return "";
    }
    // Split path
    char separator = '/'; // For Linux only
    std::vector<std::string> parsedParts;
    std::stringstream ss(oldPath);
    std::string part;

    while (getline(ss, part, separator)) {
        if (!part.empty()) {
            parsedParts.push_back(part);
        }
    }

    if (parsedParts.empty()) {
        return "";
    }

    std::string safePath = "***";
    if (parsedParts.size() > 2) { // Replace the first 2 level directories by "***"
        for (size_t i = 2; i < parsedParts.size(); ++i) {
            safePath += separator + parsedParts[i];
        }
    } else {
        safePath += separator + parsedParts.back();
    }

    return safePath;
}

static std::string GetEnvByName(const std::string &name)
{
    const char *info = std::getenv(name.c_str());
    if (info == nullptr) {
        return "";
    }
    size_t infoLength = strlen(info);
    if (infoLength > MAX_ENV_LENGTH) {
        std::cout << name << " is too long, it should be less than " << MAX_ENV_LENGTH << std::endl;
        return "";
    }
    return std::string(info);
}

bool FileUtils::GetCheckPermissionFlag()
{
    bool defaultFlag = true;
    const std::string checkPermission = GetEnvByName("MINDIE_CHECK_INPUTFILES_PERMISSION");
    if (checkPermission == "") {
        return defaultFlag;
    }
    if (checkPermission != "0" && checkPermission != "1") {
        std::cout << "Unknown permission flag " << checkPermission
                                                        << " in env MINDIE_CHECK_INPUTFILES_PERMISSION" << std::endl;
        return defaultFlag;
    }
    return checkPermission == "1";
}
} // namespace mindie_llm