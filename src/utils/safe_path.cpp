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
#include "safe_path.h"

#include <regex>
#include <unistd.h>
#include <pwd.h>
#include <grp.h>
#include <iostream>
#include <sys/stat.h>

#include "string_utils.h"
#include "safe_envvar.h"

namespace mindie_llm {

static constexpr size_t MAX_PATH_LENGTH = 4096;
static constexpr size_t MAX_LAST_NAME_LENGTH = 255;
static constexpr size_t MAX_DIR_DEPTH = 32;
static const std::string IGNORE_CHECK = "0";

static const std::map<std::string, int> modeMap = {
    {"r",   R_OK},            // 只读
    {"r+",  R_OK | W_OK},     // 读写
    {"w",   W_OK},            // 只写
    {"w+",  R_OK | W_OK},     // 读写
    {"a",   W_OK},            // 追加写
    {"a+",  R_OK | W_OK},     // 读写追加
    {"x",   X_OK}             // 只可执行
};

Result ChangePermission(const std::string& path, const fs::perms& permission)
{
    if (!fs::exists(path) || fs::is_symlink(path)) {
        return Result::Error(ResultCode::RISK_ALERT, "Path does not exist or is a symlink: " + path);
    }
    try {
        fs::permissions(path, permission);
    } catch (const fs::filesystem_error &e) {
        return Result::Error(ResultCode::IO_FAILURE, "Failed to set permissions for " + path + ": " + e.what());
    }
    return Result::OK();
}

SafePath::SafePath(std::string path,
                   PathType pathType,
                   std::string mode,
                   fs::perms maxPermission,
                   uint64_t sizeLimitation,
                   std::string suffix)
    : path_(std::move(path)),
      pathType_(pathType),
      mode_(std::move(mode)),
      maxPermission_(maxPermission),
      sizeLimitation_(sizeLimitation),
      suffix_(std::move(suffix)) {}

Result SafePath::Check(std::string& checkedPath, bool pathExist, SoftLinkLevel softLinkLevel)
{
    Result r = NormalizePath();
    if (!r.IsOk()) {
        return r;
    }
    r = pathExist ? CheckPathWhenExist(softLinkLevel) : CheckPathWhenNotExist(softLinkLevel);
    if (!r.IsOk()) {
        return r;
    }
    checkedPath = path_;
    return r;
}

Result SafePath::ExpandHome()
{
    if (path_.empty() || path_[0] != '~') {
        return Result::OK();
    }
    if (path_.size() > 1 && path_[1] != '/') {
        return Result::OK();
    }
    std::string home;
    EnvVar::GetInstance().Get("HOME", "/root/", home);
    if (path_.size() == 1) {
        path_ = home;
    } else {
        path_ = home + path_.substr(1);
    }
    return Result::OK();
}

fs::path SafePath::LongestExistingPrefix(const fs::path& abs) const
{
    fs::path current;
    for (const auto& part : abs) {
        fs::path next = current / part;
        if (fs::exists(next)) {
            current = next;
        } else {
            break;
        }
    }
    return current;
}

fs::path SafePath::LexicallyNormalize(const fs::path& path) const
{
    fs::path result;
    for (const auto& part : path) {
        if (part == ".") {
            continue;
        }
        if (part == "..") {
            if (!result.empty() &&
                result.filename() != "..") {
                result = result.parent_path();
            } else {
                result /= part;
            }
        } else {
            result /= part;
        }
    }
    return result;
}

Result SafePath::NormalizePath()
{
    Result r = ExpandHome();
    if (!r.IsOk()) {
        return r;
    }
    fs::path abs = fs::absolute(path_);
    fs::path existing = LongestExistingPrefix(abs);
    fs::path base;
    try {
        if (!existing.empty()) {
            base = fs::canonical(existing);
        } else {
            base = fs::absolute(abs.root_path());
        }
    } catch (...) {
        base = fs::absolute(abs.root_path());
    }
    fs::path remaining;
    auto abs_it = abs.begin();
    auto ex_it  = existing.begin();
    while (abs_it != abs.end() && ex_it != existing.end() && *abs_it == *ex_it) {
        ++abs_it;
        ++ex_it;
    }
    for (; abs_it != abs.end(); ++abs_it) {
        if (abs_it->empty()) {
            continue;
        }
        remaining /= *abs_it;
    }
    fs::path result = LexicallyNormalize(base / remaining);
    std::string s = result.string();
    if (s.size() > 1 && s.back() == fs::path::preferred_separator) {
        s.pop_back();
    }
    path_ = s;
    return Result::OK();
}

Result SafePath::CheckPathWhenExist(SoftLinkLevel softLinkLevel)
{
    Result r = CheckPathExist();
    if (!r.IsOk()) {
        return r;
    }
    r = CheckSpecialChars();
    if (!r.IsOk()) {
        return r;
    }
    r = CheckPathLength();
    if (!r.IsOk()) {
        return r;
    }
    r = CheckSoftLink(softLinkLevel);
    if (!r.IsOk()) {
        return r;
    }
    if (pathType_ == PathType::FILE) {
        r = IsFile();
        if (!r.IsOk()) {
            return r;
        }
        r = CheckFileSuffix();
        if (!r.IsOk()) {
            return r;
        }
        r = CheckFileSize();
        if (!r.IsOk()) {
            return r;
        }
    } else if (pathType_ == PathType::DIR) {
        r = IsDir();
        if (!r.IsOk()) {
            return r;
        }
        r = CheckDirSize();
        if (!r.IsOk()) {
            return r;
        }
        if (!IsSuffix(path_, "/")) {
            path_ += "/";
        }
    }
    return CheckPermission();
}

Result SafePath::CheckPathWhenNotExist(SoftLinkLevel softLinkLevel)
{
    Result r = CheckSpecialChars();
    if (!r.IsOk()) {
        return r;
    }
    r = CheckPathLength();
    if (!r.IsOk()) {
        return r;
    }
    r = CheckSoftLink(softLinkLevel);
    if (!r.IsOk()) {
        return r;
    }
    return Result::OK();
}

Result SafePath::IsFile() const
{
    if (!fs::is_regular_file(path_)) {
        return Result::Error(ResultCode::TYPE_MISMATCH, "The path is not file: " + path_);
    }
    return Result::OK();
}

Result SafePath::IsDir() const
{
    if (!fs::is_directory(path_)) {
        return Result::Error(ResultCode::TYPE_MISMATCH, "The path is not directory: " + path_);
    }
    return Result::OK();
}

Result SafePath::CheckPathExist() const
{
    if (!fs::exists(path_)) {
        return Result::Error(ResultCode::NONE_ARGUMENT, "Path not found: " + path_);
    }
    return Result::OK();
}

Result SafePath::CheckSoftLink(SoftLinkLevel level)
{
    if (!fs::is_symlink(path_)) {
        return Result::OK();
    }
    if (level == SoftLinkLevel::STRICT) {
        return Result::Error(ResultCode::RISK_ALERT, "Found symlink path: " + path_);
    }
    std::string resolvedPath = fs::read_symlink(path_).string();
    std::cout << "Found symlink path: " << path_ << ", using real path: " << resolvedPath << std::endl;
    path_ = resolvedPath;
    return Result::OK();
}

Result SafePath::CheckMode() const
{
    auto it = modeMap.find(mode_);
    if (it == modeMap.end()) {
        std::string keys = GetKeysFromMap(modeMap, ",");
        return Result::Error(ResultCode::INVALID_ARGUMENT,
            "Unsupported mode: " + mode_ + ". Only supported modes are: " + keys);
    }
    int accessMode = it->second;
    if (access(path_.c_str(), accessMode) != 0) {
        return Result::Error(ResultCode::NO_PERMISSION,
            "Insufficient permissions for mode '" + mode_ + "' on path: " + path_);
    }
    return Result::OK();
}

Result SafePath::CheckMaxPermission() const
{
    if (maxPermission_ == fs::perms::unknown) {
        return Result::Error(ResultCode::INVALID_ARGUMENT, "Maximum permission is not configured");
    }
    struct ::stat st;
    if (lstat(path_.c_str(), &st) != 0) {
        return Result::Error(ResultCode::NO_PERMISSION, "Cannot stat path: " + path_);
    }
    mode_t actualPerm = st.st_mode & static_cast<mode_t>(0777);
    mode_t maxPerm = static_cast<mode_t>(maxPermission_);
    uid_t current_uid = ::getuid();
    if (((actualPerm & ~maxPerm) == 0) || current_uid == 0) {
        return Result::OK();
    }
    std::ostringstream ossCheckMaxPermission;
    ossCheckMaxPermission << "Path permission exceeds maximum allowed: " << path_
                          << ", actual=0" << std::oct << actualPerm
                          << ", limit=0" << std::oct << maxPerm;
    std::string isCheckPermission;
    EnvVar::GetInstance().Get(MINDIE_CHECK_INPUTFILES_PERMISSION, DEFAULT_CHECK_PERM, isCheckPermission);
    if (isCheckPermission != IGNORE_CHECK) {
        return Result::Error(ResultCode::NO_PERMISSION, ossCheckMaxPermission.str());
    } else {
        std::cout << ossCheckMaxPermission.str()
            << ", permission check is disabled by env "
            << MINDIE_CHECK_INPUTFILES_PERMISSION
            << ", excessive permission is ignored, this may introduce security risks" << std::endl;
        return Result::OK();
    }
}

Result SafePath::CheckOwner() const
{
    struct ::stat st;
    if (lstat(path_.c_str(), &st) != 0) {
        return Result::Error(ResultCode::NO_PERMISSION, "Cannot stat path: " + path_);
    }
    uid_t current_uid = ::getuid();
    if ((current_uid == st.st_uid) || (current_uid == 0)) {
        return Result::OK();
    }
    std::ostringstream ossCheckOwner;
    ossCheckOwner << "path owner uid mismatch: path=" << path_
                  << ", current_uid=" << std::to_string(static_cast<unsigned long>(current_uid))
                  << ", path_uid=" << std::to_string(static_cast<unsigned long>(st.st_uid));
    std::string isCheckPermission;
    EnvVar::GetInstance().Get(MINDIE_CHECK_INPUTFILES_PERMISSION, DEFAULT_CHECK_PERM, isCheckPermission);
    if (isCheckPermission != IGNORE_CHECK) {
        return Result::Error(ResultCode::NO_PERMISSION, ossCheckOwner.str());
    } else {
        std::cout << ossCheckOwner.str()
            << ", permission check is disabled by env "
            << MINDIE_CHECK_INPUTFILES_PERMISSION
            << ", owner mismatch is ignored, this may introduce security risks" << std::endl;
        return Result::OK();
    }
}

Result SafePath::CheckPermission() const
{
    Result r = CheckOwner();
    if (!r.IsOk()) {
        return r;
    }
    r = CheckMode();
    if (!r.IsOk()) {
        return r;
    }
    r = CheckMaxPermission();
    if (!r.IsOk()) {
        return r;
    }
    return Result::OK();
}

Result SafePath::CheckSpecialChars() const
{
    const std::regex VALID_PATH_PATTERN(R"(^(?!.*\.\.)[a-zA-Z0-9_./-]+$)");
    if (!std::regex_match(path_, VALID_PATH_PATTERN)) {
        return Result::Error(ResultCode::INVALID_ARGUMENT, "Path contains special characters: " + path_);
    }
    return Result::OK();
}

Result SafePath::CheckPathLength() const
{
    if (path_.length() > MAX_PATH_LENGTH) {
        return Result::Error(ResultCode::RISK_ALERT,
            "Path length exceeds maximum limit: " + std::to_string(MAX_PATH_LENGTH));
    }
    size_t depth = 0;
    for (auto& subName : fs::path(path_)) {
        ++depth;
        if (depth > MAX_DIR_DEPTH) {
            return Result::Error(ResultCode::RISK_ALERT,
                "Exceeded max directory depth: " + std::to_string(MAX_DIR_DEPTH));
        }
        if (subName.string().length() > MAX_LAST_NAME_LENGTH) {
            return Result::Error(ResultCode::RISK_ALERT,
                "Directory/file name exceeds maximum length limit: " + std::to_string(MAX_LAST_NAME_LENGTH));
        }
    }
    return Result::OK();
}

Result SafePath::CheckFileSuffix() const
{
    if (!suffix_.empty() && !IsSuffix(path_, suffix_)) {
        return Result::Error(ResultCode::INVALID_ARGUMENT, path_ + " is not a " + suffix_ + " file.");
    }
    return Result::OK();
}

Result SafePath::CheckFileSize() const
{
    if (sizeLimitation_ == 0) {
        return Result::OK();
    }
    try {
        auto size = fs::file_size(path_);
        if (size > sizeLimitation_) {
            return Result::Error(ResultCode::RISK_ALERT, "File size exceeds limit: " + std::to_string(sizeLimitation_));
        }
    } catch (const fs::filesystem_error& e) {
        return Result::Error(ResultCode::IO_FAILURE, "Failed to get file size for " + path_ + ": " + e.what());
    }
    return Result::OK();
}

Result SafePath::CheckDirSize() const
{
    if (sizeLimitation_ == 0) {
        return Result::OK();
    }
    try {
        size_t totalSize = 0;
        for (auto& p : fs::recursive_directory_iterator(path_)) {
            if (fs::is_regular_file(p.path())) {
                totalSize += fs::file_size(p.path());
                if (totalSize > sizeLimitation_) {
                    return Result::Error(ResultCode::RISK_ALERT, "Directory size exceeds limit");
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        return Result::Error(ResultCode::IO_FAILURE, "Failed to traverse directory " + path_ + ": " + e.what());
    }
    return Result::OK();
}

Result MakeDirs(const std::string& pathStr)
{
    if (fs::exists(pathStr)) {
        return Result::OK();
    }
    std::string makingPath;
    SafePath checkDir(pathStr, PathType::DIR, "w", PERM_750);
    Result r = checkDir.Check(makingPath, false);
    if (!r.IsOk()) {
        return r;
    }
    try {
        fs::create_directories(makingPath);
    } catch (const fs::filesystem_error &e) {
        return Result::Error(
            ResultCode::IO_FAILURE, "Failed to create directories for " + makingPath + ": " + e.what()
        );
    }
    ChangePermission(makingPath, PERM_750);
    return Result::OK();
}

} // namespace mindie_llm
