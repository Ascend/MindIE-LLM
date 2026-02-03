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

#ifndef SAFE_ENVVAR_H
#define SAFE_ENVVAR_H

#include "safe_result.h"

namespace mindie_llm {

inline const char *MINDIE_LOG_LEVEL = "MINDIE_LOG_LEVEL";
inline const char *MINDIE_LOG_TO_FILE = "MINDIE_LOG_TO_FILE";
inline const char *MINDIE_LOG_TO_STDOUT = "MINDIE_LOG_TO_STDOUT";
inline const char *MIES_INSTALL_PATH = "MIES_INSTALL_PATH";
inline const char *MINDIE_CHECK_INPUTFILES_PERMISSION = "MINDIE_CHECK_INPUTFILES_PERMISSION";

const std::string DEFAULT_MIES_INSTALL_PATH = "/usr/local/Ascend/mindie/latest/mindie-service";
const std::string DEFAULT_CHECK_PERM = "";


class EnvVar {
public:
    static EnvVar& GetInstance();

    Result Set(const char *key, const std::string& value, bool overwrite = true) const;

    Result Get(const char *key, const std::string& defaultValue, std::string& outValue) const;

    EnvVar(const EnvVar&) = delete;
    EnvVar& operator=(const EnvVar&) = delete;

private:
    EnvVar() = default;
};

} // namespace mindie_llm

#endif
