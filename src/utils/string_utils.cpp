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

#include "string_utils.h"

#include <sstream>
#include <map>
#include <algorithm>
#include <cctype>

namespace mindie_llm {
void Split(const std::string& text, char delimiter, std::vector<std::string>& tokens)
{
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
}

void Split(const std::string& text, const std::string& delimiter, std::vector<std::string>& tokens)
{
    if (delimiter.empty()) {
        throw std::invalid_argument("Delimiter cannot be empty.");
    }

    size_t pos = 0;
    size_t prev = 0;
    while ((pos = text.find(delimiter, prev)) != std::string::npos) {
        tokens.push_back(text.substr(prev, pos - prev));
        prev = pos + delimiter.length();
    }
    if (prev < text.length()) {
        tokens.push_back(text.substr(prev));
    }
}

void RemoveSpaces(std::string& text)
{
    // 将所有空格符移动到字符串右边
    auto newEnd = std::remove_if(text.begin(), text.end(), [](unsigned char c) {
        return std::isspace(c);
    });
    text.erase(newEnd, text.end());
}

void TrimSpaces(std::string& text)
{
    // 去掉字符串前面的空格
    text.erase(text.begin(),
               std::find_if(text.begin(), text.end(),
                            [](unsigned char c) {
                                return !std::isspace(c);
                            }));
    // 去掉字符串后面的空格
    text.erase(std::find_if(text.rbegin(), text.rend(),
                            [](unsigned char c) {
                                return !std::isspace(c);
                            }).base(),
               text.end());
}

bool IsSuffix(const std::string& str, const std::string& suffix)
{
    if (str.length() < suffix.length()) {
        return false;
    }
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

void ToLower(std::string& str)
{
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::tolower(c); });
}

void ToUpper(std::string& str)
{
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::toupper(c); });
}

std::unordered_map<std::string, std::string> ParseKeyValueString(
    const std::string& input,
    const std::unordered_set<std::string>& validValues, const std::string& defaultKey,
    char pairDelimiter, char kvDelimiter)
{
    std::unordered_map<std::string, std::string> result;
    std::vector<std::string> segments;
    Split(input, pairDelimiter, segments);
    for (auto& seg : segments) {
        TrimSpaces(seg);
        ToLower(seg);
        if (seg.empty()) {
            continue;
        }
        size_t pos = seg.find(kvDelimiter);
        std::string key, value;
        if (pos != std::string::npos) {
            key = seg.substr(0, pos);
            value = seg.substr(pos + 1);
        } else {
            key = defaultKey;
            value = seg;
        }
        if (validValues.empty() || validValues.count(value) > 0) {
            result[key] = value;
        }
    }
    return result;
}

std::unordered_map<std::string, std::string>ParseArgs(const std::string& str)
{
    std::unordered_map<std::string, std::string> result;
    std::istringstream iss(str);
    std::string arg;
    std::string value;
    while (iss >> arg) {
        if (arg.empty() || arg[0] != '-') {
            throw std::runtime_error("Invalid argument: " + arg);
        }
        if (!(iss >> value)) {
            throw std::runtime_error("Missing value for argument: " + arg);
        }
        result[arg] = value;
    }
    return result;
}

void SplitTokensToVec(const std::string& text, char delimiter, std::vector<long>& tokens)
{
    std::stringstream ss(text);
    std::string token;
    try {
        while (std::getline(ss, token, delimiter)) {
            tokens.push_back(std::stol(token));
        }
    } catch (const std::exception &e) {
        throw std::invalid_argument("Invalid Token Id: " + token);
    }
}

} // namespace mindie_llm
