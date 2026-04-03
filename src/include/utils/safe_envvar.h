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
inline const char *MINDIE_LOG_PATH = "MINDIE_LOG_PATH";
inline const char *MINDIE_LOG_VERBOSE = "MINDIE_LOG_VERBOSE";
inline const char *MINDIE_LOG_ROTATE = "MINDIE_LOG_ROTATE";
inline const char *MINDIE_LLM_HOME_PATH = "MINDIE_LLM_HOME_PATH";
inline const char *MINDIE_CHECK_INPUTFILES_PERMISSION = "MINDIE_CHECK_INPUTFILES_PERMISSION";

const std::string DEFAULT_MINDIE_LOG_LEVEL = "info";
const std::string DEFAULT_MINDIE_LOG_TO_FILE = "1";
const std::string DEFAULT_MINDIE_LOG_TO_STDOUT = "0";
const std::string DEFAULT_MINDIE_LOG_PATH = "~/mindie/log/";
const std::string DEFAULT_MINDIE_LOG_VERBOSE = "1";
const std::string DEFAULT_MINDIE_LOG_ROTATE = "-fs 20 -r 10";      // Rotating log files, 20 MB each, keep 10 files.
const std::string DEFAULT_CHECK_PERM = "";

const std::string& GetDefaultMindIELLMHomePath();

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
