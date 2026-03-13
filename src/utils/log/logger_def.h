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

#ifndef MINDIE_LOG_DEF_H
#define MINDIE_LOG_DEF_H

#include <string>
#include <sstream>
namespace mindie_llm {
// model name
const std::string MINDIE_SERVER = "server";

// submodel name
const std::string SUBMODLE_NAME_DAEMON = "daemon";
const std::string SUBMODLE_NAME_ENDPOINT = "endpoint";
const std::string SUBMODLE_NAME_TOKENIZER = "tokenizer";
const std::string SUBMODLE_NAME_INFERINSTANCE = "infer_instance";
const std::string SUBMODLE_NAME_HEALTHCHECKER = "health_checker";

// ERROR code head
const std::string MINDIE_ERRORCODE_HEAD = "MIE";
// model code
const std::string MINDIE_SERVER_CODE = "04";

// error level
const std::string INFO = "I";
const std::string WARNING = "W";
const std::string ERROR = "E";
const std::string CRITICAL = "C";

// submodel code
const std::string SUBMODLE_CODE_DAEMON = "01";
const std::string SUBMODLE_CODE_ENDPOINT = "02";
const std::string SUBMODLE_CODE_TOKENIZER = "04";
const std::string SUBMODLE_CODE_INFERINSTANCE = "06";
const std::string SUBMODLE_CODE_HEALTHCHECKER = "07";

// features code
// 服务初始化
const std::string SUBMODLE_FEATURE_INIT = "01";
// 业务请求
const std::string SUBMODLE_FEATURE_SERVER_REQUEST = "02";
// 业务请求
const std::string SUBMODLE_FEATURE_MANAGE_REQUEST = "03";
// 单机推理
const std::string SUBMODLE_FEATURE_SINGLE_INFERENCE = "04";
// 多机推理
const std::string SUBMODLE_FEATURE_MULTI_INFERENCE = "05";
// tokenizer
const std::string SUBMODLE_FEATURE_TOKENIZER = "06";
// detokenizer
const std::string SUBMODLE_FEATURE_DETOKENIZER = "07";
// response响应处理
const std::string SUBMODLE_FEATURE_RESPONSE = "08";
// PD分离splitwise
const std::string SUBMODLE_FEATURE_SPLITWISE = "09";
// splitfuse
const std::string SUBMODLE_FEATURE_SPLITFUSE = "0A";
// prefix cache
const std::string SUBMODLE_FEATURE_PREFIX_CACHE = "0B";
// 抢占
const std::string SUBMODLE_FEATURE_OCCUPYING = "0C";
// recompute
const std::string SUBMODLE_FEATURE_RECOMPUTE = "0D";
// 调度策略fcfs等
const std::string SUBMODLE_FEATURE_SCHEDULER_POLICY = "0E";
// profiling
const std::string SUBMODLE_FEATURE_PROFILING = "0F";
// 热配置
const std::string SUBMODLE_FEATURE_HOT_CONFIG = "10";
// 安全相关
const std::string SUBMODLE_FEATURE_SECURE = "11";
// 故障控制恢复
const std::string SUBMODLE_FEATURE_FAULT_CONTROL = "12";

// ERROR type code
// 权限异常
const std::string PERMISSION_ERROR = "01";
// 子进程异常
const std::string SUBPROCESS_ERROR = "02";
// 推理服务拉起异常
const std::string STARTUP_ERROR = "03";
// 请求参数解析异常
const std::string PARAM_PARSE_ERROR = "04";
// 推理请求生成异常
const std::string INFERENCE_GENERATE_REQUEST_ERROR = "05";
// 状态告警
const std::string STATUS_WARNING = "06";
// 校验异常
const std::string CHECK_ERROR = "07";
// 组件调用异常
const std::string LOCAL_INVOKING_ERROR = "08";
// 库调用异常
const std::string SYSTEM_INVOKING_ERROR = "09";
// tensor添加/获取异常
const std::string TENSOR_ERROR = "0A";
// encode/decode转换异常
const std::string ENCODE_DECODE_ERROR = "0B";
// 请求响应处理异常
const std::string RESPONSE_PROCESS_ERROR = "0C";
// 请求响应生成异常
const std::string RESPONSE_GENERATE_ERROR = "0D";
// json解析异常
const std::string JSON_PARSE_ERROR = "0E";
// 超时告警
const std::string TIMEOUT_WARNING = "0F";
// 空响应告警
const std::string EMPTY_RESPONSE_WARNING = "10";
// pull kv 异常
const std::string PULL_KV_ERROR = "11";
// 重计算 异常
const std::string RECOMPUTE_ERROR = "12";
// 网络安全相关异常
const std::string SECURITY_ERROR = "13";
// 异常传递
const std::string ABNORMAL_TRANSMISSION_ERROR = "14";
// 校验告警
const std::string CHECK_WARNING = "15";
// 未知异常
const std::string UNKNOWN_ERROR = "16";
// 下载异常
const std::string DOWNLOAD_ERROR = "17";
// 删除异常
const std::string REMOVE_ERROR = "18";
// 等待子进程告警
const std::string WATTING_SUBPROCESS_WARNING = "19";
// 子进程退出告警
const std::string EXIT_SUBPROCESS_WARNING = "1A";
// 配置校验异常
const std::string CONFIG_ERROR = "1B";
// 初始化异常
const std::string INIT_ERROR = "1C";
// 状态正常
const std::string SIMULATE_NORMAL = "20";

// err code = ERROR code head + model code + error level + submodel code + features code + ERROR type code

inline std::string GenerateDaemonErrCode(const std::string &level, const std::string &featType,
                                         const std::string &errType)
{
    std::stringstream ss;
    ss << "[" + MINDIE_ERRORCODE_HEAD + MINDIE_SERVER_CODE + level + SUBMODLE_CODE_DAEMON + featType + errType + "] ";
    return ss.str();
}

inline std::string GenerateEndpointErrCode(const std::string &level, const std::string &featType,
                                           const std::string &errType)
{
    std::stringstream ss;
    ss << "[" + MINDIE_ERRORCODE_HEAD + MINDIE_SERVER_CODE + level + SUBMODLE_CODE_ENDPOINT + featType + errType + "] ";
    return ss.str();
}

inline std::string GenerateTokenizerErrCode(const std::string &level, const std::string &featType,
                                            const std::string &errType)
{
    std::stringstream ss;
    ss << "[" + MINDIE_ERRORCODE_HEAD + MINDIE_SERVER_CODE + level + SUBMODLE_CODE_TOKENIZER + featType + errType +
              "] ";
    return ss.str();
}
inline std::string GenerateInferInstanceErrCode(const std::string &level, const std::string &featType,
    const std::string &errType)
{
    std::stringstream ss;
    ss << "[" + MINDIE_ERRORCODE_HEAD + MINDIE_SERVER_CODE + level + SUBMODLE_CODE_INFERINSTANCE + featType + errType +
              "] ";
    return ss.str();
}

inline std::string GenerateHealthCheckerErrCode(
    const std::string& level,
    const std::string& featType,
    const std::string& errType)
{
    return "[" + MINDIE_ERRORCODE_HEAD + MINDIE_SERVER_CODE + level +
        SUBMODLE_CODE_HEALTHCHECKER + featType + errType + "] ";
}
} // namespace mindie_llm
#endif