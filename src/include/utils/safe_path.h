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

#ifndef SAFE_PATH_H
#define SAFE_PATH_H

#include <experimental/filesystem>
#include <string>
#include <vector>
#include <optional>

#include "safe_result.h"

namespace mindie_llm {

namespace fs = std::experimental::filesystem;
inline constexpr fs::perms PERM_750 = fs::perms::owner_all | fs::perms::group_read | fs::perms::group_exec;
inline constexpr fs::perms PERM_640 = fs::perms::owner_read | fs::perms::owner_write | fs::perms::group_read;
inline constexpr fs::perms PERM_440 = fs::perms::owner_read | fs::perms::group_read;
inline constexpr size_t SIZE_1MB = 1024 * 1024;
inline constexpr size_t SIZE_20MB = 20 * 1024 * 1024;
inline constexpr size_t SIZE_500MB = 500 * 1024 * 1024;

Result ChangePermission(const std::string& path, const fs::perms& permission);
Result MakeDirs(const std::string& pathStr);

enum class PathType {
    FILE,
    DIR
};

enum class SoftLinkLevel {
    IGNORE,
    STRICT
};

class SafePath {
public:
    SafePath(std::string path,
             PathType pathType,
             std::string mode,
             fs::perms maxPermission,
             uint64_t sizeLimitation = 0,
             std::string suffix = "");

    Result Check(std::string& checkedPath, bool pathExist = true, SoftLinkLevel softLinkLevel = SoftLinkLevel::STRICT);

private:
    std::string path_;
    PathType pathType_;
    std::string mode_;
    fs::perms maxPermission_;
    uint64_t sizeLimitation_;
    std::string suffix_;

    Result ExpandHome();
    fs::path LexicallyNormalize(const fs::path& path) const;
    fs::path LongestExistingPrefix(const fs::path& abs) const;
    Result NormalizePath();
    Result CheckPathWhenExist(SoftLinkLevel softLinkLevel);
    Result CheckPathWhenNotExist(SoftLinkLevel softLinkLevel);

    Result CheckPathExist() const;
    Result IsFile() const;
    Result IsDir() const;
    Result CheckSoftLink(SoftLinkLevel level);
    Result CheckOwner() const;
    Result CheckMaxPermission() const;
    Result CheckPermission() const;
    Result CheckMode() const;
    Result CheckSpecialChars() const;
    Result CheckPathLength() const;
    Result CheckFileSuffix() const;
    Result CheckFileSize() const;
    Result CheckDirSize() const;
};

} // namespace mindie_llm

#endif
