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
 
#include <gtest/gtest.h>
#include <string>
#include <climits>
#include <unistd.h>
#include <set>
#include <vector>
#include <map>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <common_util.h>
#include <chrono>
#include <thread>
#include <nlohmann/json.hpp>
using Json = nlohmann::json;

namespace mindie_llm {

// Test Suite for CanonicalPath function
class CommonUtilsTest : public ::testing::Test {
public:
    std::string configPath_;
    const char* mindieLlmPath;
    const char* mindieMiesPath;

protected:
    void SetUp() override
    {
        setenv("MINDIE_LLM_HOME_PATH", MINDIE_LLM_HOME_PATH_TEST, 1);
        const char* mindieLlmPathEnv = getenv("MINDIE_LLM_HOME_PATH");
        if (mindieLlmPathEnv != nullptr) {
            mindieLlmPath = mindieLlmPathEnv;
        }

        const char* mindieMiesPathEnv = getenv("MINDIE_LLM_HOME_PATH");
        if (mindieMiesPathEnv != nullptr) {
            mindieMiesPath = mindieMiesPathEnv;
        }
        SetConfigPath();
    }

    void TearDown() override
    {
    }

    void SetConfigPath()
    {
        std::string homePath;
        GetLlmPath(homePath);
        configPath_ = homePath + "/conf/config.json";
    }
};

// GetDuration测试
// 测试正常持续时间
TEST_F(CommonUtilsTest, TestGetDurationSuccess)
{
    auto start = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));  // 暂停 100 毫秒
    auto end = std::chrono::steady_clock::now();

    size_t duration = GetDuration(end, start);
    EXPECT_GE(duration, static_cast<size_t>(100));  // 持续时间应该大于等于 100 毫秒
}

// GetCurTime测试
// 测试获取时间正确
TEST_F(CommonUtilsTest, TestGetCurTimeSuccess)
{
    std::string timeStr = GetCurTime();

    auto now = std::chrono::system_clock::now();
    std::time_t nowC = std::chrono::system_clock::to_time_t(now);
    tm *parts = std::localtime(&nowC);

    std::stringstream expectedTime;
    expectedTime << std::put_time(parts, "%Y-%m-%d %H:%M:%S");
    constexpr size_t dataStart = 0;
    constexpr size_t dataEnd = 10;
    constexpr size_t timeStart = 11;
    constexpr size_t timeEnd = 5;
    // 确保返回的时间字符串与系统时间相近（通常在几秒内）

    EXPECT_EQ(timeStr.substr(dataStart, dataEnd), expectedTime.str().substr(dataStart, dataEnd));
    EXPECT_EQ(timeStr.substr(timeStart, timeEnd), expectedTime.str().substr(timeStart, timeEnd));
}

// Split测试
// 测试字符串分割正确
TEST_F(CommonUtilsTest, TestSplitByBasicStrSuccess)
{
    std::string splitStr = "llm,service";
    char delim = ',';
    std::vector<std::string> result = Split(splitStr, delim);
    constexpr size_t expectedSize = 2;
    ASSERT_EQ(result.size(), static_cast<size_t>(expectedSize));
    EXPECT_EQ(result[0], "llm");
    EXPECT_EQ(result[1], "service");
}

// 测试单个字符串分割正确
TEST_F(CommonUtilsTest, TestSplitBySingleStrSuccess)
{
    std::string splitStr = "llm";
    char delim = ',';
    std::vector<std::string> result = Split(splitStr, delim);
    ASSERT_EQ(result.size(), static_cast<size_t>(1));
    EXPECT_EQ(result[0], "llm");
}

// 测试空字符串分割正确
TEST_F(CommonUtilsTest, TestSplitByEmptyStrSuccess)
{
    std::string splitStr = "";
    char delim = ',';
    std::vector<std::string> result = Split(splitStr, delim);
    ASSERT_EQ(result.size(), static_cast<size_t>(0));
}

// CanonicalPath测试
// 测试空路径
TEST_F(CommonUtilsTest, TestCanonicalPathByEmptyPathFail)
{
    std::string path = "";
    EXPECT_FALSE(CanonicalPath(path));
}

// 测试有效绝对路径
TEST_F(CommonUtilsTest, TestCanonicalByValidPathSuccess)
{
    const char* homePath = getenv("HOME");
    if (homePath != nullptr) {
        std::string path = homePath;
        bool result = CanonicalPath(path);
        EXPECT_TRUE(result);
        EXPECT_EQ(path, homePath);
    }
}

// 测试路径长度超出限制
TEST_F(CommonUtilsTest, TestCanonicalByPathTooLongFail)
{
    std::string path(PATH_MAX + 1, 'a');  // 创建一个超长路径
    EXPECT_FALSE(CanonicalPath(path));  // 应该返回 false
}

// GetHomePath测试
// 测试是否能获取正确路径
TEST_F(CommonUtilsTest, TestGetHomePathSuccess)
{
    std::string homePath;
    Error result = GetHomePath(homePath);
    EXPECT_TRUE(result.IsOk());
}

// GetLlmPath测试
// 测试是否能获取正确路径
TEST_F(CommonUtilsTest, TestGetLlmPathSuccess)
{
    std::string llmPath;
    Error result = GetLlmPath(llmPath);
    EXPECT_TRUE(result.IsOk());
}

// 测试错误的环境变量是否被校验住
TEST_F(CommonUtilsTest, TestGetLlmPathByInvalidEnvFail)
{
    setenv("MINDIE_LLM_HOME_PATH", "", 1);
    std::string llmPath;
    Error result = GetLlmPath(llmPath);
    setenv("MINDIE_LLM_HOME_PATH", mindieLlmPath, 1);
    EXPECT_FALSE(result.IsOk());
}

// IsNumber测试
TEST_F(CommonUtilsTest, TestIsNumberByEmptyStringFail)
{
    EXPECT_FALSE(IsNumber(""));
}

TEST_F(CommonUtilsTest, TestIsNumberByValidNumberSuccess)
{
    EXPECT_TRUE(IsNumber("123"));
    EXPECT_TRUE(IsNumber("-123"));
    EXPECT_TRUE(IsNumber("0"));
}

TEST_F(CommonUtilsTest, TestIsNumberByInvalidNumberFail)
{
    EXPECT_FALSE(IsNumber("123a"));
    EXPECT_FALSE(IsNumber("-123a"));
    EXPECT_FALSE(IsNumber("12 3"));
    EXPECT_FALSE(IsNumber(" 123"));
}

// GetConfigPath测试
TEST_F(CommonUtilsTest, TestGetConfigPathWithoutEnvSuccess)
{
    std::string outConfigPath;
    Error result = GetConfigPath(outConfigPath);
    EXPECT_TRUE(result.IsOk());
}

// 设置环境变量MIES_CONFIG_JSON_PATH测试
TEST_F(CommonUtilsTest, TestGetConfigPathWithEnvSuccess)
{
    setenv("MIES_CONFIG_JSON_PATH", configPath_.c_str(), 1);
    std::string outConfigPath;
    Error result = GetConfigPath(outConfigPath);
    unsetenv("MIES_CONFIG_JSON_PATH");
    EXPECT_TRUE(result.IsOk());
}

// GetModelInfo测试
TEST_F(CommonUtilsTest, TestGetModelInfoSuccess)
{
    std::string modelName;
    size_t serverCount = 0;
    size_t tp = 0;
    GetModelInfo(configPath_, modelName, tp, serverCount);
    EXPECT_EQ(modelName, "llama_65b");
}

// CheckSystemConfig测试
TEST_F(CommonUtilsTest, TestCheckSystemConfigSuccess)
{
    Json backendJsonData;
    bool result = CheckSystemConfig(configPath_, backendJsonData, "BackendConfig");
    EXPECT_TRUE(result);
}

// TrimSpace测试
TEST_F(CommonUtilsTest, TestTrimSpaceSuccess)
{
    // 测试前后都有空格
    EXPECT_EQ(mindie_llm::TrimSpace("  hello world  "), "hello world");

    // 测试只有前面有空格
    EXPECT_EQ(mindie_llm::TrimSpace("  hello"), "hello");

    // 测试只有后面有空格
    EXPECT_EQ(mindie_llm::TrimSpace("hello  "), "hello");

    // 测试没有空格
    EXPECT_EQ(mindie_llm::TrimSpace("hello"), "hello");

    // 测试全是空格
    EXPECT_EQ(mindie_llm::TrimSpace("   "), "");

    // 测试空字符串
    EXPECT_EQ(mindie_llm::TrimSpace(""), "");
}

// ToLower测试
TEST_F(CommonUtilsTest, TestToLowerSuccess)
{
    EXPECT_EQ(mindie_llm::ToLower("HELLO WORLD"), "hello world");
    EXPECT_EQ(mindie_llm::ToLower("Hello123"), "hello123");
    EXPECT_EQ(mindie_llm::ToLower("hello"), "hello");
    EXPECT_EQ(mindie_llm::ToLower(""), "");
}

// ToUpper测试
TEST_F(CommonUtilsTest, TestToUpperSuccess)
{
    EXPECT_EQ(mindie_llm::ToUpper("hello world"), "HELLO WORLD");
    EXPECT_EQ(mindie_llm::ToUpper("Hello123"), "HELLO123");
    EXPECT_EQ(mindie_llm::ToUpper("HELLO"), "HELLO");
    EXPECT_EQ(mindie_llm::ToUpper(""), "");
}

// GetHostIP测试
TEST_F(CommonUtilsTest, TestGetHostIPSuccess)
{
    std::vector<std::string> ips = mindie_llm::GetHostIP(true);
    // 应该至少有一个IP地址
    EXPECT_FALSE(ips.empty());

    // 测试不跳过回环地址
    std::vector<std::string> ipsWithLoopback = mindie_llm::GetHostIP(false);
    EXPECT_FALSE(ipsWithLoopback.empty());
}

// GetBinaryPath测试
TEST_F(CommonUtilsTest, TestGetBinaryPathSuccess)
{
    std::string binaryPath;
    bool result = mindie_llm::GetBinaryPath(binaryPath);
    EXPECT_TRUE(result);
    EXPECT_FALSE(binaryPath.empty());
}

// JoinStrings测试
TEST_F(CommonUtilsTest, TestJoinStringsSuccess)
{
    std::vector<std::string> strings = {"hello", "world", "test"};
    std::string result = mindie_llm::JoinStrings(strings, ",");
    EXPECT_EQ(result, "hello,world,test");

    // 测试空向量
    std::vector<std::string> emptyStrings;
    std::string emptyResult = mindie_llm::JoinStrings(emptyStrings, ",");
    EXPECT_EQ(emptyResult, "");

    // 测试单个元素
    std::vector<std::string> singleString = {"hello"};
    std::string singleResult = mindie_llm::JoinStrings(singleString, ",");
    EXPECT_EQ(singleResult, "hello");
}

// RandomNumber测试
TEST_F(CommonUtilsTest, TestRandomNumberSuccess)
{
    uint32_t maxNumber = 100;
    uint32_t randomNum = mindie_llm::RandomNumber(maxNumber);
    EXPECT_LE(randomNum, maxNumber);

    // 测试边界情况
    uint32_t zeroResult = mindie_llm::RandomNumber(0);
    EXPECT_EQ(zeroResult, 0);
}

// SerializeSet测试
TEST_F(CommonUtilsTest, TestSerializeSetSuccess)
{
    std::set<uint32_t> testSet = {1, 2, 3, 5, 8};
    std::string serialized = mindie_llm::SerializeSet(testSet);
    EXPECT_EQ(serialized, "1,2,3,5,8");

    // 测试空集合
    std::set<uint32_t> emptySet;
    std::string emptySerialized = mindie_llm::SerializeSet(emptySet);
    EXPECT_EQ(emptySerialized, "");

    // 测试单个元素
    std::set<uint32_t> singleSet = {42};
    std::string singleSerialized = mindie_llm::SerializeSet(singleSet);
    EXPECT_EQ(singleSerialized, "42");
}

// DeserializeSet测试
TEST_F(CommonUtilsTest, TestDeserializeSetSuccess)
{
    std::string data = "1,2,3,5,8";
    std::set<size_t> result = mindie_llm::DeserializeSet(data);
    std::set<size_t> expected = {1, 2, 3, 5, 8};
    EXPECT_EQ(result, expected);

    // 测试空字符串
    std::set<size_t> emptyResult = mindie_llm::DeserializeSet("");
    EXPECT_TRUE(emptyResult.empty());

    // 测试单个元素
    std::set<size_t> singleResult = mindie_llm::DeserializeSet("42");
    std::set<size_t> singleExpected = {42};
    EXPECT_EQ(singleResult, singleExpected);
}

// ParsePortFromIp测试
TEST_F(CommonUtilsTest, TestParsePortFromIpSuccess)
{
    uint32_t port;
    bool result = mindie_llm::ParsePortFromIp("192.168.1.1;8080", port);
    EXPECT_TRUE(result);
    EXPECT_EQ(port, 8080);

    // 测试没有端口的情况
    bool noPortResult = mindie_llm::ParsePortFromIp("192.168.1.1", port);
    EXPECT_FALSE(noPortResult);
}

// ReverseDpInstId测试
TEST_F(CommonUtilsTest, TestReverseDpInstIdSuccess)
{
    uint64_t dpInstanceId = 1234500067; // pid=123450, dpIdx=67
    auto result = mindie_llm::ReverseDpInstId(dpInstanceId);
    EXPECT_EQ(result.first, 123450);   // pid
    EXPECT_EQ(result.second, 67);     // dpIdx
}

// CleanStringForJson测试
TEST_F(CommonUtilsTest, TestCleanStringForJsonSuccess)
{
    // 测试正常字符串
    std::string normalStr = "Hello World";
    std::string cleaned = mindie_llm::CleanStringForJson(normalStr);
    EXPECT_EQ(cleaned, "Hello World");

    // 测试包含控制字符的字符串
    std::string controlStr = "Hello\x01World\x02";
    std::string controlCleaned = mindie_llm::CleanStringForJson(controlStr);
    EXPECT_EQ(controlCleaned, "HelloWorld");

    // 测试包含换行符的字符串
    std::string newlineStr = "Hello\nWorld\r\nTest";
    std::string newlineCleaned = mindie_llm::CleanStringForJson(newlineStr);
    EXPECT_EQ(newlineCleaned, "Hello\nWorld\r\nTest");

    // 测试空字符串
    std::string emptyCleaned = mindie_llm::CleanStringForJson("");
    EXPECT_EQ(emptyCleaned, "");
}

// IsFloatEquals测试
TEST_F(CommonUtilsTest, TestIsFloatEqualsSuccess)
{
    EXPECT_TRUE(mindie_llm::IsFloatEquals(1.0f, 1.0f));
    EXPECT_TRUE(mindie_llm::IsFloatEquals(1.0f, 1.000001f));
    EXPECT_FALSE(mindie_llm::IsFloatEquals(1.0f, 1.1f));
    EXPECT_TRUE(mindie_llm::IsFloatEquals(0.0f, 0.0f));
    EXPECT_TRUE(mindie_llm::IsFloatEquals(-1.0f, -1.0f));
}

// SplitString测试
TEST_F(CommonUtilsTest, TestSplitStringSuccess)
{
    std::string testStr = "hello,world,test";
    std::vector<std::string> result = mindie_llm::SplitString(testStr, ',');
    std::vector<std::string> expected = {"hello", "world", "test"};
    EXPECT_EQ(result, expected);

    // 测试空字符串
    std::vector<std::string> emptyResult = mindie_llm::SplitString("", ',');
    EXPECT_TRUE(emptyResult.empty());

    // 测试没有分隔符的字符串
    std::vector<std::string> noDelimResult = mindie_llm::SplitString("hello", ',');
    std::vector<std::string> noDelimExpected = {"hello"};
    EXPECT_EQ(noDelimResult, noDelimExpected);
}

// SplitPath测试
TEST_F(CommonUtilsTest, TestSplitPathSuccess)
{
    std::string path = "/usr/local/bin";
    std::vector<std::string> result = mindie_llm::SplitPath(path);
    std::vector<std::string> expected = {"usr", "local", "bin"};
    EXPECT_EQ(result, expected);

    // 测试根路径
    std::vector<std::string> rootResult = mindie_llm::SplitPath("/");
    EXPECT_TRUE(rootResult.empty());

    // 测试相对路径
    std::vector<std::string> relativeResult = mindie_llm::SplitPath("usr/local");
    std::vector<std::string> relativeExpected = {"usr", "local"};
    EXPECT_EQ(relativeResult, relativeExpected);
}

// AbsoluteToAnonymousPath测试
TEST_F(CommonUtilsTest, TestAbsoluteToAnonymousPathSuccess)
{
    // 测试正常路径
    std::string path = "/usr/local/bin";
    std::string result = mindie_llm::AbsoluteToAnonymousPath(path);
    EXPECT_EQ(result, "/******/******/bin");

    // 测试相对路径
    std::string relativePath = "usr/local/bin";
    std::string relativeResult = mindie_llm::AbsoluteToAnonymousPath(relativePath);
    EXPECT_EQ(relativeResult, "");

    // 测试根路径
    std::string rootResult = mindie_llm::AbsoluteToAnonymousPath("/");
    EXPECT_EQ(rootResult, "/");

    // 测试单级路径
    std::string singleResult = mindie_llm::AbsoluteToAnonymousPath("/usr");
    EXPECT_EQ(singleResult, "/******");
}

// AbsoluteToRelativePath测试
TEST_F(CommonUtilsTest, TestAbsoluteToRelativePathSuccess)
{
    // 测试正常情况
    std::string absPath = "/usr/local/bin/test";
    std::string absDir = "/usr/local";
    std::string result = mindie_llm::AbsoluteToRelativePath(absPath, absDir);
    EXPECT_EQ(result, "/******/******/bin/test");

    // 测试空路径
    std::string emptyResult = mindie_llm::AbsoluteToRelativePath("", "/usr");
    EXPECT_TRUE(emptyResult.empty());

    // 测试不匹配的目录
    std::string mismatchResult = mindie_llm::AbsoluteToRelativePath("/usr/local/bin", "/opt");
    EXPECT_TRUE(mismatchResult.find("******") != std::string::npos);
}

// 测试模板函数 VectorToString
TEST_F(CommonUtilsTest, TestVectorToStringSuccess)
{
    std::vector<int> intVec = {1, 2, 3, 4, 5};
    std::string result = mindie_llm::VectorToString(intVec);
    EXPECT_EQ(result, "[1, 2, 3, 4, 5]");

    std::vector<std::string> strVec = {"hello", "world"};
    std::string strResult = mindie_llm::VectorToString(strVec);
    EXPECT_EQ(strResult, "[hello, world]");

    // 测试空向量
    std::vector<int> emptyVec;
    std::string emptyResult = mindie_llm::VectorToString(emptyVec);
    EXPECT_EQ(emptyResult, "[]");
}

// 测试模板函数 MapToString
TEST_F(CommonUtilsTest, TestMapToStringSuccess)
{
    std::map<int, std::string> testMap = {{1, "one"}, {2, "two"}};
    std::string result = mindie_llm::MapToString(testMap);
    EXPECT_TRUE(result.find("1: one") != std::string::npos);
    EXPECT_TRUE(result.find("2: two") != std::string::npos);

    std::map<int, std::vector<std::string>> complexMap = {
        {1, {"a", "b"}},
        {2, {"c", "d"}}
    };
    std::string complexResult = mindie_llm::MapToString(complexMap);
    EXPECT_TRUE(complexResult.find("1: [a, b]") != std::string::npos);
    EXPECT_TRUE(complexResult.find("2: [c, d]") != std::string::npos);
}

// 测试模板函数 MergeMaps
TEST_F(CommonUtilsTest, TestMergeMapsSuccess)
{
    std::map<int, int> totalMap = {{1, 10}, {2, 20}};
    std::map<int, int> subMap = {{1, 5}, {3, 30}};

    mindie_llm::MergeMaps(totalMap, subMap);
    EXPECT_EQ(totalMap[1], 15);  // 10 + 5
    EXPECT_EQ(totalMap[2], 20);  // 保持不变
    EXPECT_EQ(totalMap[3], 30);  // 新增
}

// 测试模板函数 RemoveMapElements
TEST_F(CommonUtilsTest, TestRemoveMapElementsSuccess)
{
    std::map<int, std::string> inputMap = {{1, "one"}, {2, "two"}, {3, "three"}};
    std::vector<int> keysToRemove = {1, 3};

    std::map<int, std::string> result = mindie_llm::RemoveMapElements(inputMap, keysToRemove);
    EXPECT_EQ(result.size(), 1);
    EXPECT_TRUE(result.find(2) != result.end());
    EXPECT_EQ(result[2], "two");
}

// CheckIPV4测试
TEST_F(CommonUtilsTest, TestCheckIPV4Success)
{
    // 测试有效的IPv4地址
    EXPECT_TRUE(mindie_llm::CheckIPV4("192.168.1.1", "testIP", true));
    EXPECT_TRUE(mindie_llm::CheckIPV4("10.0.0.1", "testIP", true));
    EXPECT_TRUE(mindie_llm::CheckIPV4("172.16.0.1", "testIP", true));
    EXPECT_TRUE(mindie_llm::CheckIPV4("127.0.0.1", "testIP", true));
    EXPECT_TRUE(mindie_llm::CheckIPV4("255.255.255.255", "testIP", true));
    
    // 测试0.0.0.0地址（enableZeroIp=true时允许）
    EXPECT_TRUE(mindie_llm::CheckIPV4("0.0.0.0", "testIP", true));
    
    // 测试0.0.0.0地址（enableZeroIp=false时不允许）
    EXPECT_FALSE(mindie_llm::CheckIPV4("0.0.0.0", "testIP", false));
}

TEST_F(CommonUtilsTest, TestCheckIPV4Fail)
{
    // 测试无效的IPv4地址
    EXPECT_FALSE(mindie_llm::CheckIPV4("256.1.2.3", "testIP", true));
    EXPECT_FALSE(mindie_llm::CheckIPV4("1.2.3.256", "testIP", true));
    EXPECT_FALSE(mindie_llm::CheckIPV4("192.168.1", "testIP", true));
    EXPECT_FALSE(mindie_llm::CheckIPV4("192.168.1.1.1", "testIP", true));
    EXPECT_FALSE(mindie_llm::CheckIPV4("abc.def.ghi.jkl", "testIP", true));
    EXPECT_FALSE(mindie_llm::CheckIPV4("192.168.001.001", "testIP", true));
    
    // 测试空字符串
    EXPECT_FALSE(mindie_llm::CheckIPV4("", "testIP", true));
    
    // 测试超长地址
    std::string longIP(33, '1'); // 超过MAX_IPV4_LENGTH
    longIP[3] = longIP[7] = longIP[11] = longIP[15] = '.';
    EXPECT_FALSE(mindie_llm::CheckIPV4(longIP, "testIP", true));
}

// CheckIPV6测试
TEST_F(CommonUtilsTest, TestCheckIPV6Success)
{
    // 测试有效的IPv6地址
    EXPECT_TRUE(mindie_llm::CheckIPV6("::1", "testIP", true));
    EXPECT_TRUE(mindie_llm::CheckIPV6("2001:db8::1", "testIP", true));
    EXPECT_TRUE(mindie_llm::CheckIPV6("2001:db8:0:0:0:0:0:1", "testIP", true));
    EXPECT_TRUE(mindie_llm::CheckIPV6("2001:db8::1:0:0:0:1", "testIP", true));
    EXPECT_TRUE(mindie_llm::CheckIPV6("[2001:db8::1]", "testIP", true));
    
    // 测试::地址（enableZeroIp=true时允许）
    EXPECT_TRUE(mindie_llm::CheckIPV6("::", "testIP", true));
    
    // 测试::地址（enableZeroIp=false时不允许）
    EXPECT_FALSE(mindie_llm::CheckIPV6("::", "testIP", false));
}

TEST_F(CommonUtilsTest, TestCheckIPV6Fail)
{
    // 测试无效的IPv6地址
    EXPECT_FALSE(mindie_llm::CheckIPV6("2001:db8::1::1", "testIP", true));
    EXPECT_FALSE(mindie_llm::CheckIPV6("2001:db8:g::1", "testIP", true));
    EXPECT_FALSE(mindie_llm::CheckIPV6("2001:db8::1:", "testIP", true));
    EXPECT_FALSE(mindie_llm::CheckIPV6(":2001:db8::1", "testIP", true));
    EXPECT_FALSE(mindie_llm::CheckIPV6("2001:db8::1:1:1:1:1:1:1", "testIP", true));
    
    // 测试空字符串
    EXPECT_FALSE(mindie_llm::CheckIPV6("", "testIP", true));
    
    // 测试超长地址
    std::string longIP(129, '1'); // 超过MAX_IPV6_LENGTH
    EXPECT_FALSE(mindie_llm::CheckIPV6(longIP, "testIP", true));
}

// CheckIp测试
TEST_F(CommonUtilsTest, TestCheckIpSuccess)
{
    // 测试有效的IPv4地址
    EXPECT_TRUE(mindie_llm::CheckIp("192.168.1.1", "testIP", true));
    EXPECT_TRUE(mindie_llm::CheckIp("10.0.0.1", "testIP", true));
    EXPECT_TRUE(mindie_llm::CheckIp("127.0.0.1", "testIP", true));
    
    // 测试有效的IPv6地址
    EXPECT_TRUE(mindie_llm::CheckIp("::1", "testIP", true));
    EXPECT_TRUE(mindie_llm::CheckIp("2001:db8::1", "testIP", true));
    EXPECT_TRUE(mindie_llm::CheckIp("[2001:db8::1]", "testIP", true));
    
    // 测试0地址（enableZeroIp=true时允许）
    EXPECT_TRUE(mindie_llm::CheckIp("0.0.0.0", "testIP", true));
    EXPECT_TRUE(mindie_llm::CheckIp("::", "testIP", true));
    
    // 测试0地址（enableZeroIp=false时不允许）
    EXPECT_FALSE(mindie_llm::CheckIp("0.0.0.0", "testIP", false));
    EXPECT_FALSE(mindie_llm::CheckIp("::", "testIP", false));
}

TEST_F(CommonUtilsTest, TestCheckIpFail)
{
    EXPECT_FALSE(mindie_llm::CheckIp("", "testIP", true));
    
    // 测试既不是IPv4也不是IPv6的地址
    EXPECT_FALSE(mindie_llm::CheckIp("localhost", "testIP", true));
    EXPECT_FALSE(mindie_llm::CheckIp("example.com", "testIP", true));
    EXPECT_FALSE(mindie_llm::CheckIp("192.168.1", "testIP", true));
    EXPECT_FALSE(mindie_llm::CheckIp("2001:db8", "testIP", true));
    
    // 测试无效的IPv4地址
    EXPECT_FALSE(mindie_llm::CheckIp("256.1.2.3", "testIP", true));
    EXPECT_FALSE(mindie_llm::CheckIp("192.168.001.001", "testIP", true));
    
    // 测试无效的IPv6地址
    EXPECT_FALSE(mindie_llm::CheckIp("2001:db8::1::1", "testIP", true));
    EXPECT_FALSE(mindie_llm::CheckIp("2001:db8:g::1", "testIP", true));
}

TEST_F(CommonUtilsTest, TestCheckIpEdgeCases)
{
    // 测试混合格式（包含冒号和点，但格式不正确）
    EXPECT_FALSE(mindie_llm::CheckIp("192.168.1:8080", "testIP", true));
    EXPECT_FALSE(mindie_llm::CheckIp("2001:db8.1.2.3", "testIP", true));
    
    // 测试特殊字符
    EXPECT_FALSE(mindie_llm::CheckIp("192.168.1.1@", "testIP", true));
    EXPECT_FALSE(mindie_llm::CheckIp("2001:db8::1#", "testIP", true));
    
    // 测试前导和尾随空格
    EXPECT_FALSE(mindie_llm::CheckIp(" 192.168.1.1", "testIP", true));
    EXPECT_FALSE(mindie_llm::CheckIp("192.168.1.1 ", "testIP", true));
    EXPECT_FALSE(mindie_llm::CheckIp(" 2001:db8::1 ", "testIP", true));
    
    // 测试端口号格式（应该被拒绝）
    EXPECT_FALSE(mindie_llm::CheckIp("192.168.1.1:8080", "testIP", true));
    EXPECT_FALSE(mindie_llm::CheckIp("[2001:db8::1]:8080", "testIP", true));
}
}
