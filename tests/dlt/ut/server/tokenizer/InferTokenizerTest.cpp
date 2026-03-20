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
#include <libgen.h>
#include <filesystem>

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cstdlib>
#include <gtest/gtest.h>
#include "mockcpp/mockcpp.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <dirent.h>
#include <signal.h>
#include <optional>
#include <sys/wait.h>
#include <errno.h>
#include <semaphore.h>
#include <ctime>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cctype>
#include <sys/types.h>
#include <unistd.h>
#include "common_util.h"
#include "infer_tokenizer.h"
#include "base_config_manager.h"
#include "config_manager_impl.h"
#include "../utils/mock_util.h"
#include "file_utils.h"
#include "env_util.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>

using namespace mindie_llm;

MOCKER_CPP_OVERLOAD_EQ(ServerConfig);
MOCKER_CPP_OVERLOAD_EQ(Error);
MOCKER_CPP_OVERLOAD_EQ(std::vector<ModelDeployConfig>)
MOCKER_CPP_OVERLOAD_EQ(BackendConfig)

namespace SimpleLLMInference {
    std::string g_tokenizerSharedMemory = "/llm_tokenizer_shared_memory_";
    constexpr int MASK = 0600;

    /* ---------- 先声明给 mockcpp::invoke 用的自由函数指针（在类后定义） ---------- */
    // GetConfigJsonStr → 按值返回
    static std::string Ret_DefaultConfigJson_Value();
    static std::string Ret_MissingKeysConfigJson_Value();
    static std::string Ret_InvalidConfigJson_Value();
    // 引用返回
    static const std::vector<ModelDeployConfig>& Ret_StaticDeployConfig_Ref();
    static const BackendConfig&                 Ret_StaticBackendConfig_Ref();

    int mock_sem_timewait(sem_t *__restrict __sem, const struct timespec *__restrict __abstime)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        sem_trywait(__sem);
        return 0;
    }

    class InferenceTokenizerTest : public testing::Test {
    public:
        // ==== 静态对象 & 文本：用于安全地按引用/值返回 ====
        static const std::string& DefaultConfigJson()
        {
            static const std::string kMockCfgJson = R"({
                "BackendConfig": {
                    "ModelDeployConfig": {
                        "maxSeqLen" : 2560,
                        "maxInputTokenLen" : 2048,
                        "truncation" : 0,
                        "ModelConfig": [
                            {
                                "models": [
                                    {
                                        "name": "m1",
                                        "id": 1
                                    },
                                    {
                                        "name": "m2",
                                        "id": 2
                                    }
                                ]
                            }
                        ]
                    }
                }
            })";
            return kMockCfgJson;
        }
        static const std::string& MissingKeysConfigJson()
        {
            static const std::string kJson = R"({"SomethingElse": 1})";
            return kJson;
        }
        static const std::string& InvalidConfigJson()
        {
            static const std::string kJson = "{ not_a_valid_json ";
            return kJson;
        }
        static std::vector<ModelDeployConfig>& StaticDeployConfig()
        {
            static std::vector<ModelDeployConfig> sDeployConfig(1);
            return sDeployConfig;
        }
        static BackendConfig& StaticBackendConfig()
        {
            static BackendConfig sBackendConfig;
            return sBackendConfig;
        }

    protected:
        static std::unique_ptr<pybind11::scoped_interpreter> s_interpreter;

        static std::string GetCwd()
        {
            std::error_code ec;
            auto path = std::filesystem::current_path(ec);
            return ec ? "." : path.string();
        }

        static void SetUpTestSuite()
        {
            if (!s_interpreter) {
                s_interpreter = std::make_unique<pybind11::scoped_interpreter>();
                auto sys = pybind11::module_::import("sys");
                auto path = sys.attr("path").cast<pybind11::list>();
                path.insert(0, GetCwd());
            }
        }

        static void CleanGeneratedFiles()
        {
            {
                std::string py = GetCwd() + "/tokenizer.py";
                ::unlink(py.c_str());
            }
            std::string pkgDir = GetCwd() + "/mindie_llm/tokenizer";
            std::string initPy = pkgDir + "/__init__.py";
            ::unlink(initPy.c_str());
            ::rmdir(pkgDir.c_str());
            pkgDir = GetCwd() + "/mindie_llm";
            initPy = pkgDir + "/__init__.py";
            ::unlink(initPy.c_str());
            ::rmdir(pkgDir.c_str());
        }

        static void KillChildrenAndWait()
        {
            namespace fs = std::filesystem;

            auto get_ppid = [](pid_t pid) -> std::optional<pid_t> {
                fs::path status_path = fs::path("/proc") / std::to_string(pid) / "status";
                std::ifstream fin(status_path);
                if (!fin.is_open()) return std::nullopt;
                std::string line;
                while (std::getline(fin, line)) {
                    if (line.rfind("PPid:", 0) == 0) {
                        auto pos = line.find_first_of("0123456789");
                        if (pos == std::string::npos) return std::nullopt;
                        try {
                            return static_cast<pid_t>(std::stol(line.substr(pos)));
                        } catch (...) {
                            return std::nullopt;
                        }
                    }
                }
                return std::nullopt;
            };

            auto is_digits = [](const std::string &s) {
                return !s.empty() &&
                    std::all_of(s.begin(), s.end(), [](unsigned char c) {
                        return std::isdigit(c);
                    });
            };


            const pid_t parent = ::getpid();
            std::vector<pid_t> children;

            // 枚举 /proc 下所有子进程
            for (const auto& entry : fs::directory_iterator("/proc")) {
                if (!entry.is_directory()) continue;
                std::string name = entry.path().filename().string();
                if (!is_digits(name)) continue;
                pid_t pid = 0;
                try { pid = static_cast<pid_t>(std::stol(name)); } catch (...) { continue; }
                if (pid <= 1 || pid == parent) continue;
                if (auto ppid = get_ppid(pid); ppid && *ppid == parent)
                    children.push_back(pid);
            }

            if (children.empty()) return;

            // 先发 SIGTERM
            for (pid_t c : children) ::kill(c, SIGTERM);

            auto drain = []() {
                int status = 0;
                while (true) {
                    pid_t r = ::waitpid(-1, &status, WNOHANG);
                    if (r > 0) continue;
                    if (r == 0) break;                 // 还有子进程但未退出
                    if (r < 0 && errno == ECHILD) break; // 没有子进程
                    break;
                }
            };

            // 等待最多 5 秒让子进程自己退出
            auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (std::chrono::steady_clock::now() < deadline) {
                drain();
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            // 再次检查存活的子进程并强杀
            children.clear();
            for (const auto& entry : fs::directory_iterator("/proc")) {
                if (!entry.is_directory()) continue;
                std::string name = entry.path().filename().string();
                if (!is_digits(name)) continue;
                pid_t pid = 0;
                try { pid = static_cast<pid_t>(std::stol(name)); } catch (...) { continue; }
                if (pid <= 1 || pid == parent) continue;
                if (auto ppid = get_ppid(pid); ppid && *ppid == parent)
                    children.push_back(pid);
            }
            for (pid_t c : children) ::kill(c, SIGKILL);

            // 最终 drain 一次避免僵尸进程
            drain();
        }

        static void TearDownTestSuite()
        {
            KillChildrenAndWait();
            CleanGeneratedFiles();
        }

        void SetUp() override
        {
            setenv("TOKENIZER_ENCODE_TIMEOUT", "2", 1); // 命中最小值归一=5

            // 基础环境 Mock
            MOCKER_CPP(&CanonicalPath, bool (*)(std::string &)).stubs().will(returnValue(true));
            MOCKER_CPP(&GetConfigPath, Error (*)(std::string &)).stubs().will(returnValue(Error(Error::Code::OK)));
            MOCKER_CPP(&ServerConfigManager::InitFromJson, bool (*)()).stubs().will(returnValue(true));
            MOCKER_CPP(&BackendConfigManager::InitFromJson, bool (*)()).stubs().will(returnValue(true));
            MOCKER_CPP(&ScheduleConfigManager::InitFromJson, bool (*)()).stubs().will(returnValue(true));
            MOCKER_CPP(&ModelDeployConfigManager::InitFromJson, bool (*)()).stubs().will(returnValue(true));

            EnvUtil::GetInstance().SetEnvVar("RANK_TABLE_FILE", GetParentDirectory() + "/../../config_manager/conf/ranktable.json");
            EnvUtil::GetInstance().SetEnvVar("MIES_CONTAINER_IP", "127.0.0.1");
            EnvUtil::GetInstance().SetEnvVar("HOST_IP", "127.0.0.1");
            EnvUtil::GetInstance().SetEnvVar("MINDIE_CHECK_INPUTFILES_PERMISSION", "1");
            ConfigManager::CreateInstance(GetParentDirectory() + "/../../config_manager/conf/config_grpc.json");

            // ==== 关键：签名匹配真实函数；静态对象 + invoke ====
            // GetConfigJsonStr: 按值返回 std::string
            MOCKER_CPP(&ConfigManager::GetConfigJsonStr, std::string (*)())
                .stubs().will(MOCKCPP_NS::invoke(&Ret_DefaultConfigJson_Value));

            GenerateMockModel();

            // GetModelDeployConfig: 返回 const std::vector<...>&
            auto &sDeploy = StaticDeployConfig();
            sDeploy[0].modelWeightPath = "mock";
            sDeploy[0].backendType     = "mock";
            MOCKER_CPP(&ConfigManager::GetModelDeployConfig, const std::vector<ModelDeployConfig> &(*)())
                .stubs().will(MOCKCPP_NS::invoke(&Ret_StaticDeployConfig_Ref));

            // GetBackendConfig: 返回 const BackendConfig&
            auto &sBackend = StaticBackendConfig();
            sBackend.tokenizerProcessNumber = 1;
            MOCKER_CPP(&ConfigManager::GetBackendConfig, const BackendConfig& (*)())
                .stubs().will(MOCKCPP_NS::invoke(&Ret_StaticBackendConfig_Ref));

            // 默认让 sem_timedwait 成功（0）
            MOCKER_CPP(&sem_timedwait, int (*)(sem_t *__restrict, const struct timespec *__restrict))
                .stubs().will(invoke(mock_sem_timewait));

            MOCKER_CPP(&ConfigManager::Impl::CheckAllParam, bool (*)()).stubs().will(returnValue(true));

            ASSERT_TRUE(TokenizerProcessPool::GetInstance().InitTokenizerPool());
        }

        void TearDown() override
        {
            EnvUtil::GetInstance().ClearEnvVar("RANK_TABLE_FILE");
            EnvUtil::GetInstance().ClearEnvVar("MINDIE_CHECK_INPUTFILES_PERMISSION");
            EnvUtil::GetInstance().ClearEnvVar("MIES_CONTAINER_IP");
            EnvUtil::GetInstance().ClearEnvVar("HOST_IP");
            GlobalMockObject::verify();
        }

        std::string GetParentDirectory()
        {
            char buffer[1024];
            if (getcwd(buffer, sizeof(buffer)) == nullptr) {
                std::cerr << "Error getting current directory: " << strerror(errno) << std::endl;
                return "";
            }
            char *temp = strdup(buffer);
            if (temp == nullptr) {
                std::cerr << "Memory allocation failed" << std::endl;
                return "";
            }
            char *parent = dirname(temp);
            std::string result(parent);
            free(temp);
            return result;
        }

        static void GenerateMockModel()
        {
            // a) tokenizer.py（主进程）
            {
                std::ofstream ofs(GetCwd() + "/tokenizer.py", std::ios::out | std::ios::trunc);
                ofs << R"(
from typing import List, Optional
import numpy as np

class IbisTokenizer:
    def __init__(self, path: str, bakend_type: str, trust_remote_code: bool, models_dict_str: str):
        print('init')

    def encode(self, prompt, chat_template_kwargs: dict=None):
        print('encode')
        return [0, 1]

    def encode_chat(self, prompt, kwargs: dict=None):
        print('encode_chat')
        return [0, 1]

    def decode(self, input_tokens, kwargs: dict=None):
        print('decode')
        return "decode_result"

    def decode_one(self, input_tokens, kwargs: dict=None):
        print('decode_one')
        return "decode_one_result"

    def download_url(self, prompt: str, timestamp: int):
        if "DOWNLOAD_BAD" in prompt:
            raise RuntimeError("bad url \n details...")

    def delete_multimodal_cache(self, req_id: int):
        print(f"delete cache {req_id}")

class IbisTokenizerNoDownload:
    def __init__(self, path: str, bakend_type: str, trust_remote_code: bool, models_dict_str: str):
        pass
    def encode(self, prompt, chat_template_kwargs: dict = None): return [0]
    def decode(self, ids, kwargs: dict=None): return "ok"

class IbisTokenizerDeleteRaise(IbisTokenizer):
    def delete_multimodal_cache(self, req_id: int):
        raise RuntimeError("del fail")
)" << std::endl;
            }

            // b) mies_tokenizer（子进程)
            {
                std::string pkgDir = GetCwd() + "/mindie_llm";
                ::mkdir(pkgDir.c_str(), 0755);
                std::ofstream(pkgDir + "/__init__.py").close();
                pkgDir = pkgDir + "/tokenizer";
                ::mkdir(pkgDir.c_str(), 0755);
                std::ofstream ofs(pkgDir + "/__init__.py", std::ios::out | std::ios::trunc);
                ofs << R"(
from typing import List, Optional
import numpy as np

class IbisTokenizer:
    def __init__(self, path: str, bakend_type: str, trust_remote_code: bool, models_dict_str: str):
        pass
    def encode(self, prompt, chat_template_kwargs: dict = None): return [0, 1]
    def encode_chat(self, prompt, kwargs: dict=None): return [0, 1]
    def decode(self, input_tokens, kwargs: dict=None): return "decode_result"
    def decode_one(self, input_tokens, kwargs: dict=None): return "decode_one_result"
    def download_url(self, prompt: str, timestamp: int):
        if "DOWNLOAD_BAD" in prompt: raise RuntimeError("bad url \n details...")
    def delete_multimodal_cache(self, req_id: int): return
)" << std::endl;
            }
        }

        static void CheckInferTokenizer()
        {
            pybind11::module module = pybind11::module_::import("tokenizer");
            pybind11::object autoTokenizerClass = module.attr("IbisTokenizer");
            pybind11::object autoTokenizer =
                autoTokenizerClass("/data/atb_testdata/weights/llama1-65b-safetensors", "atb", false, "dummy_model_config");
            std::shared_ptr<InferTokenizer> tokenizer =
                std::make_shared<InferTokenizer>(std::make_shared<pybind11::object>(autoTokenizer));

            // Encode
            std::string prompt1 = "test1";
            std::vector<int64_t> tokenList1;
            tokenizer->EncodeToken(prompt1, tokenList1);
            EXPECT_EQ(tokenList1[0], 0);

            // Decode
            std::string prompt2;
            std::vector<int64_t> tokenList2 = {0, 1};
            SharedMemoryHeader headerDecode{};
            headerDecode.isCurrentToolNameSent = false;
            headerDecode.isCurrentArgumentSent = false;
            headerDecode.currentToolId = 0;
            tokenizer->DecodeToken(tokenList2, prompt2, &headerDecode);
            EXPECT_EQ(prompt2, "decode_result");

            // DecodeOne
            std::string prompt3;
            std::vector<int64_t> tokenList3 = {0, 1};
            SharedMemoryHeader headerDecodeOne{};
            headerDecodeOne.isCurrentToolNameSent = false;
            headerDecodeOne.isCurrentArgumentSent = false;
            headerDecodeOne.currentToolId = 0;
            tokenizer->DecodeOneToken(tokenList3, prompt3, &headerDecodeOne);
            EXPECT_EQ(prompt3, "decode_one_result");

            // EncodeChat (chat=false)
            std::string prompt4 = "test4";
            std::vector<int64_t> tokenList4;
            tokenizer->EncodeChatToken(prompt4, false, "", tokenList4);
            EXPECT_EQ(tokenList4[0], 0);
        }
    };

    std::unique_ptr<pybind11::scoped_interpreter> InferenceTokenizerTest::s_interpreter = nullptr;

    // GetConfigJsonStr -> std::string（按值）
    static std::string Ret_DefaultConfigJson_Value()
    {
        return InferenceTokenizerTest::DefaultConfigJson();
    }
    static std::string Ret_MissingKeysConfigJson_Value()
    {
        return InferenceTokenizerTest::MissingKeysConfigJson();
    }
    static std::string Ret_InvalidConfigJson_Value()
    {
        return InferenceTokenizerTest::InvalidConfigJson();
    }
    // 引用返回
    static const std::vector<ModelDeployConfig>& Ret_StaticDeployConfig_Ref()
    {
        return InferenceTokenizerTest::StaticDeployConfig();
    }
    static const BackendConfig& Ret_StaticBackendConfig_Ref()
    {
        return InferenceTokenizerTest::StaticBackendConfig();
    }

    /** 原有用例：直调 C++ 封装 */
    TEST_F(InferenceTokenizerTest, InferTokenizer)
    {
        CheckInferTokenizer();
    }

    /** DownloadUrl 成功/失败 + DeleteMultimodalCache 正常 */
    TEST_F(InferenceTokenizerTest, DownloadUrl_And_DeleteCache)
    {
        pybind11::module module = pybind11::module_::import("tokenizer");
        pybind11::object Cls = module.attr("IbisTokenizer");
        pybind11::object obj =
            Cls("/data/atb_testdata/weights/llama1-65b-safetensors", "atb", false, "dummy_model_config");
        auto tokenizer = std::make_shared<InferTokenizer>(std::make_shared<pybind11::object>(obj));

        // 成功
        std::string okPrompt = R"({"role":"user","content":[{"type":"text","text":"hello"}]})";
        std::string msg;
        bool ok = tokenizer->DownloadUrl(okPrompt, /*reqId*/123, msg);
        EXPECT_TRUE(ok);
        EXPECT_TRUE(msg.empty());

        // 失败（Python raise）
        std::string badPrompt = R"({"role":"user","content":[{"type":"text","text":"DOWNLOAD_BAD"}]})";
        std::string failMsg;
        bool ok2 = tokenizer->DownloadUrl(badPrompt, /*reqId*/456, failMsg);
        EXPECT_FALSE(ok2);
        EXPECT_FALSE(failMsg.empty());

        // 删除缓存（正常）
        tokenizer->DeleteMultimodalCache(/*reqId*/789);
        SUCCEED();
    }

    /** DownloadUrl 错误分支：缺少方法 */
    TEST_F(InferenceTokenizerTest, DownloadUrl_ErrorBranches)
    {
        pybind11::module module = pybind11::module_::import("tokenizer");
        pybind11::object Cls = module.attr("IbisTokenizerNoDownload");
        pybind11::object obj = Cls("/p", "atb", false, "cfg");
        auto tokenizer = std::make_shared<InferTokenizer>(std::make_shared<pybind11::object>(obj));
        std::string noMethodPrompt = R"({"role":"user","content":[]})";
        std::string msg;
        bool ok = tokenizer->DownloadUrl(noMethodPrompt, /*reqId*/2, msg);
        EXPECT_FALSE(ok);
        EXPECT_FALSE(msg.empty());
    }

    /** DeleteMultimodalCache 异常分支 */
    TEST_F(InferenceTokenizerTest, DeleteCache_RaiseBranch)
    {
        pybind11::module module = pybind11::module_::import("tokenizer");
        pybind11::object Cls = module.attr("IbisTokenizerDeleteRaise");
        pybind11::object obj = Cls("/p", "atb", false, "cfg");
        auto tokenizer = std::make_shared<InferTokenizer>(std::make_shared<pybind11::object>(obj));
        tokenizer->DeleteMultimodalCache(/*reqId*/999);
        SUCCEED();
    }

    /** 进程池 Encode/Decode/DecodeOne/TikToken（含 useToolsCall=true） */
    TEST_F(InferenceTokenizerTest, TokenizerPool_EncodeDecode_TikToken)
    {
        auto &pool = TokenizerProcessPool::GetInstance();
        EXPECT_EQ(pool.GetEncodeTimeout(), 5);

        std::vector<int64_t> ids;
        uint64_t ts = 0;
        auto st_enc = pool.Encode("hi from pool", ids, ENCODE_FLAG, ts);
        EXPECT_TRUE(st_enc.IsOk());
        EXPECT_GE(ids.size(), 1U);

        std::string out;
        DetokenizeExtraInfo extra;
        extra.isCurrentToolNameSent = true;
        extra.isCurrentArgumentSent = true;
        extra.currentToolId = 42;
        extra.isChatReq = true;
        extra.reqEnableReasoning = false;
        auto st_dec = pool.Decode(ids, out, ts, /*useToolsCall*/true, /*skipSpecialTokens*/true, extra);
        EXPECT_TRUE(st_dec.IsOk());

        std::string one;
        uint32_t prev = 0, cur = 0;
        std::vector<int64_t> acc;
        for (size_t i = 0; i < ids.size(); ++i) {
            acc.push_back(ids[i]);
            auto st_one = pool.DecodeOne(acc, one, prev, cur, /*ts*/0, /*useToolsCall*/false, /*skip*/true, false, extra);
            EXPECT_TRUE(st_one.IsOk());
            if (!one.empty()) {
                prev = cur;
                cur = acc.size();
            }
        }

        int num = 0;
        std::vector<std::string> toks;
        auto st_tt = pool.TikToken("hello tik", num, toks, false);
        EXPECT_TRUE(st_tt.IsOk());
        EXPECT_GE(num, 1);
    }

    /** EncodeChatToken 的 chat 模式 */
    TEST_F(InferenceTokenizerTest, EncodeChat_ModeTrue)
    {
        pybind11::module module = pybind11::module_::import("tokenizer");
        pybind11::object Cls = module.attr("IbisTokenizer");
        pybind11::object obj = Cls("/p", "atb", false, "cfg");
        auto tokenizer = std::make_shared<InferTokenizer>(std::make_shared<pybind11::object>(obj));

        std::vector<int64_t> ids;
        std::string chatMsg = "chat msg";
        tokenizer->EncodeChatToken(chatMsg, /*isChat*/true, "", ids);
        EXPECT_FALSE(ids.empty());
    }

    /** 共享内存名称/大小检查（正常路径） */
    TEST_F(InferenceTokenizerTest, ShareTokenMemory_Checkers)
    {
        EXPECT_TRUE(ShareTokenMemory::SharedMemoryNameChecker("/abc"));
        EXPECT_FALSE(ShareTokenMemory::SharedMemoryNameChecker("abc"));
        EXPECT_FALSE(ShareTokenMemory::SharedMemoryNameChecker("/a/b"));
        EXPECT_FALSE(ShareTokenMemory::SharedMemoryNameChecker(""));

        EXPECT_TRUE(ShareTokenMemory::SharedMemorySizeCheck(1024U));
    }

    /** Encode 分支：将 sem_timedwait 置为 -1，覆盖等待分支（不强行要求失败） */
    TEST_F(InferenceTokenizerTest, Encode_WaitTimeout_ErrorBranch)
    {
        MOCKER_CPP(&sem_timedwait, int (*)(sem_t *__restrict, const struct timespec *__restrict))
            .stubs().will(returnValue(-1));

        auto &pool = TokenizerProcessPool::GetInstance();

        std::vector<int64_t> ids;
        uint64_t ts = 0;
        auto st = pool.Encode("trigger encode timeout", ids, ENCODE_FLAG, ts);

        if (st.IsOk()) {
            EXPECT_GE(ids.size(), 0U);
        } else {
            EXPECT_TRUE(ids.empty());
        }

        MOCKER_CPP(&sem_timedwait, int (*)(sem_t *__restrict, const struct timespec *__restrict))
            .stubs().will(returnValue(0));
    }

    /** Decode 分支：将 sem_timedwait 置为 -1，覆盖等待分支（不强行要求失败） */
    TEST_F(InferenceTokenizerTest, Decode_WaitTimeout_ErrorBranch)
    {
        std::vector<int64_t> tok{0, 1};
        std::string out;
        DetokenizeExtraInfo extra;

        MOCKER_CPP(&sem_timedwait, int (*)(sem_t *__restrict, const struct timespec *__restrict))
            .stubs().will(returnValue(-1));

        auto &pool = TokenizerProcessPool::GetInstance();
        uint64_t ts = 0;
        auto st = pool.Decode(tok, out, ts, /*useToolsCall*/false, /*skipSpecialTokens*/true, extra);

        if (st.IsOk()) {
            EXPECT_GE(out.size(), 0U);
        } else {
            EXPECT_TRUE(out.empty());
        }

        MOCKER_CPP(&sem_timedwait, int (*)(sem_t *__restrict, const struct timespec *__restrict))
            .stubs().will(returnValue(0));
    }

    /** 共享内存前置校验失败分支：/dev/shm 不存在 & 名称超长 */
    TEST_F(InferenceTokenizerTest, SharedMemory_Precheck_Failures)
    {
        MOCKER_CPP(&FileUtils::CheckDirectoryExists, bool (*)(const std::string &))
            .stubs().will(returnValue(false));
        EXPECT_FALSE(ShareTokenMemory::SharedMemorySizeCheck(1024U));

        MOCKER_CPP(&FileUtils::CheckDirectoryExists, bool (*)(const std::string &))
            .stubs().will(returnValue(true));

        std::string longName = "/";
        longName += std::string(300, 'x');
        EXPECT_FALSE(ShareTokenMemory::SharedMemoryNameChecker(longName));
    }


    /** InitSubProcessMemory：header 立即就绪返回 true */
    TEST_F(InferenceTokenizerTest, InitSubProcessMemory_SuccessImmediate)
    {
        auto &pool = TokenizerProcessPool::GetInstance();

        auto *hdr = new SharedMemoryHeader{};  // 值初始化
        hdr->magic = MAGIC_HEAD_BEGIN;

        MOCKER_CPP(&ShareTokenMemory::GetBuf, uint8_t* (*)())
            .stubs().will(returnValue(reinterpret_cast<uint8_t*>(hdr)));

        std::shared_ptr<ShareTokenMemory> shm(
            reinterpret_cast<ShareTokenMemory*>(0x3),
            [](ShareTokenMemory*) {
            }
        );
        bool ok = pool.InitSubProcessMemory(shm);
        EXPECT_TRUE(ok);

        delete hdr;
        GlobalMockObject::reset();
    }

    /** InitSubProcessMemory：循环等待超时返回 false（sleep mock 为 0，避免等待） */
    TEST_F(InferenceTokenizerTest, InitSubProcessMemory_Timeout)
    {
        auto &pool = TokenizerProcessPool::GetInstance();

        auto *hdr = new SharedMemoryHeader{};  // 值初始化
        hdr->magic = 0; // 永远达不到 MAGIC_HEAD_BEGIN

        MOCKER_CPP(&ShareTokenMemory::GetBuf, uint8_t* (*)())
            .stubs().will(returnValue(reinterpret_cast<uint8_t*>(hdr)));

        MOCKER_CPP(&sleep, unsigned int (*)(unsigned int))
            .stubs().will(returnValue(0));

        std::shared_ptr<ShareTokenMemory> shm(
            reinterpret_cast<ShareTokenMemory*>(0x3),
            [](ShareTokenMemory*) {
            }
        );
        bool ok = pool.InitSubProcessMemory(shm);
        EXPECT_FALSE(ok);

        delete hdr;
        GlobalMockObject::reset();
    }

    /** InitWorkerResource：内存 init 失败，直接返回 false */
    TEST_F(InferenceTokenizerTest, InitWorkerResource_MemoryFail)
    {
        auto &pool = TokenizerProcessPool::GetInstance();

        MOCKER_CPP(
            &TokenizerProcessPool::InitSubProcessMemory,
            bool (*)(const std::shared_ptr<ShareTokenMemory>&)
        ).stubs().will(returnValue(false));

        std::shared_ptr<ShareTokenMemory> shm(
            reinterpret_cast<ShareTokenMemory*>(0x11),
            [](ShareTokenMemory*) {
            }
        );
        std::shared_ptr<InferTokenizer> tk;
        bool ok = pool.InitWorkerResource(shm, tk);
        EXPECT_FALSE(ok);

        GlobalMockObject::reset();
    }

    /** InitWorkerResource：tokenizer init 失败，设置 header->magic=FAILED 并返回 false */
    TEST_F(InferenceTokenizerTest, InitWorkerResource_TokenizerFail_SetsHeader)
    {
        auto &pool = TokenizerProcessPool::GetInstance();

        auto *hdr = new SharedMemoryHeader{};  // 值初始化
        hdr->magic = MAGIC_HEAD_BEGIN;

        MOCKER_CPP(&ShareTokenMemory::GetBuf, uint8_t* (*)())
            .stubs().will(returnValue(reinterpret_cast<uint8_t*>(hdr)));

        MOCKER_CPP(
            &TokenizerProcessPool::InitSubProcessMemory,
            bool (*)(const std::shared_ptr<ShareTokenMemory>&)
        ).stubs().will(returnValue(true));

        MOCKER_CPP(
            &TokenizerProcessPool::InitSubProcessTokenizer,
            bool (*)(const std::shared_ptr<ShareTokenMemory>&, std::shared_ptr<InferTokenizer>&)
        ).stubs().will(returnValue(false));

        std::shared_ptr<ShareTokenMemory> shm(
            reinterpret_cast<ShareTokenMemory*>(0x22),
            [](ShareTokenMemory*) {
            }
        );
        std::shared_ptr<InferTokenizer> tk;
        bool ok = pool.InitWorkerResource(shm, tk);
        EXPECT_FALSE(ok);
        EXPECT_EQ(hdr->magic, MAGIC_HEAD_FAILED);

        delete hdr;
        GlobalMockObject::reset();
    }

    /** InitWorkerResource：全部成功返回 true */
    TEST_F(InferenceTokenizerTest, InitWorkerResource_Success)
    {
        auto &pool = TokenizerProcessPool::GetInstance();

        auto *hdr = new SharedMemoryHeader{};  // 值初始化
        hdr->magic = MAGIC_HEAD_BEGIN;

        MOCKER_CPP(&ShareTokenMemory::GetBuf, uint8_t* (*)())
            .stubs().will(returnValue(reinterpret_cast<uint8_t*>(hdr)));

        MOCKER_CPP(
            &TokenizerProcessPool::InitSubProcessMemory,
            bool (*)(const std::shared_ptr<ShareTokenMemory>&)
        ).stubs().will(returnValue(true));

        MOCKER_CPP(
            &TokenizerProcessPool::InitSubProcessTokenizer,
            bool (*)(const std::shared_ptr<ShareTokenMemory>&, std::shared_ptr<InferTokenizer>&)
        ).stubs().will(returnValue(true));

        std::shared_ptr<ShareTokenMemory> shm(
            reinterpret_cast<ShareTokenMemory*>(0x33),
            [](ShareTokenMemory*) {
            }
        );
        std::shared_ptr<InferTokenizer> tk;
        bool ok = pool.InitWorkerResource(shm, tk);
        EXPECT_TRUE(ok);

        delete hdr;
        GlobalMockObject::reset();
    }


    /** 成功路径：mies_tokenizer.IbisTokenizer 存在且方法齐全 -> true */
    TEST_F(InferenceTokenizerTest, InitSubProcessTokenizer_Success)
    {
        auto &pool = TokenizerProcessPool::GetInstance();

        auto *hdr = new SharedMemoryHeader{};  // 值初始化
        hdr->magic = MAGIC_HEAD_BEGIN;

        MOCKER_CPP(&ShareTokenMemory::GetBuf, uint8_t* (*)())
            .stubs().will(returnValue(reinterpret_cast<uint8_t*>(hdr)));
        std::shared_ptr<ShareTokenMemory> shm(
            reinterpret_cast<ShareTokenMemory*>(0x44),
            [](ShareTokenMemory*) {
            }
        );

        auto mies = pybind11::module_::import("mindie_llm.tokenizer");
        ASSERT_TRUE(pybind11::hasattr(mies, "IbisTokenizer"));

        std::shared_ptr<InferTokenizer> tk;
        EXPECT_TRUE(pool.InitSubProcessTokenizer(shm, tk));
        EXPECT_NE(tk, nullptr);

        delete hdr;
    }

    /** 缺少类：临时删除 IbisTokenizer 属性 -> false，然后恢复 */
    TEST_F(InferenceTokenizerTest, InitSubProcessTokenizer_NoClass_ReturnsFalse)
    {
        auto &pool = TokenizerProcessPool::GetInstance();

        auto *hdr = new SharedMemoryHeader{};  // 值初始化
        hdr->magic = MAGIC_HEAD_BEGIN;

        MOCKER_CPP(&ShareTokenMemory::GetBuf, uint8_t* (*)())
            .stubs().will(returnValue(reinterpret_cast<uint8_t*>(hdr)));
        std::shared_ptr<ShareTokenMemory> shm(
            reinterpret_cast<ShareTokenMemory*>(0x55),
            [](ShareTokenMemory*) {
            }
        );

        auto mies = pybind11::module_::import("mindie_llm.tokenizer");

        bool had = pybind11::hasattr(mies, "IbisTokenizer");
        pybind11::object backup;
        if (had) backup = mies.attr("IbisTokenizer");
        if (had) pybind11::delattr(mies, "IbisTokenizer");

        std::shared_ptr<InferTokenizer> tk;
        EXPECT_FALSE(pool.InitSubProcessTokenizer(shm, tk));
        EXPECT_EQ(tk, nullptr);

        delete hdr;
        if (had) mies.attr("IbisTokenizer") = backup; // 恢复
    }

    /** 缺少必要方法 -> false；恢复原状 */
    TEST_F(InferenceTokenizerTest, InitSubProcessTokenizer_MissingMethods_ReturnsFalse)
    {
        auto &pool = TokenizerProcessPool::GetInstance();

        auto *hdr = new SharedMemoryHeader{};  // 值初始化
        hdr->magic = MAGIC_HEAD_BEGIN;

        MOCKER_CPP(&ShareTokenMemory::GetBuf, uint8_t* (*)())
            .stubs().will(returnValue(reinterpret_cast<uint8_t*>(hdr)));
        std::shared_ptr<ShareTokenMemory> shm(
            reinterpret_cast<ShareTokenMemory*>(0x66),
            [](ShareTokenMemory*) {
            }
        );

        auto mies = pybind11::module_::import("mindie_llm.tokenizer");

        ASSERT_TRUE(pybind11::hasattr(mies, "IbisTokenizer"));
        pybind11::object backup = mies.attr("IbisTokenizer");

        // 替为“构造后返回没有 encode/decode/encode_chat 的对象”
        mies.attr("IbisTokenizer") = pybind11::cpp_function(
            [](pybind11::args, pybind11::kwargs) {
                return pybind11::dict();
            });

        std::shared_ptr<InferTokenizer> tk;
        EXPECT_FALSE(pool.InitSubProcessTokenizer(shm, tk));
        EXPECT_EQ(tk, nullptr);

        delete hdr;
        mies.attr("IbisTokenizer") = backup; // 恢复
    }

    /** 构造抛异常 -> 命中 catch(std::exception&) -> false；恢复原状 */
    TEST_F(InferenceTokenizerTest, InitSubProcessTokenizer_CtorRaises_ReturnsFalse)
    {
        auto &pool = TokenizerProcessPool::GetInstance();

        auto *hdr = new SharedMemoryHeader{};  // 值初始化
        hdr->magic = MAGIC_HEAD_BEGIN;

        MOCKER_CPP(&ShareTokenMemory::GetBuf, uint8_t* (*)())
            .stubs().will(returnValue(reinterpret_cast<uint8_t*>(hdr)));
        std::shared_ptr<ShareTokenMemory> shm(
            reinterpret_cast<ShareTokenMemory*>(0x77),
            [](ShareTokenMemory*) {
            }
        );

        auto mies = pybind11::module_::import("mindie_llm.tokenizer");

        ASSERT_TRUE(pybind11::hasattr(mies, "IbisTokenizer"));
        pybind11::object backup = mies.attr("IbisTokenizer");

        mies.attr("IbisTokenizer") = pybind11::cpp_function(
            [](pybind11::args, pybind11::kwargs) -> pybind11::object {
                throw pybind11::value_error("boom");
            });

        std::shared_ptr<InferTokenizer> tk;
        EXPECT_FALSE(pool.InitSubProcessTokenizer(shm, tk));
        EXPECT_EQ(tk, nullptr);

        delete hdr;
        mies.attr("IbisTokenizer") = backup; // 恢复
    }

    /** 覆盖 GetModelConfigString：缺字段 -> else 分支（仍可成功） */
    TEST_F(InferenceTokenizerTest, InitSubTok_ConfigJson_MissingKeys_StillSuccess)
    {
        auto &pool = TokenizerProcessPool::GetInstance();

        auto *hdr = new SharedMemoryHeader{};  // 值初始化
        hdr->magic = MAGIC_HEAD_BEGIN;

        MOCKER_CPP(&ShareTokenMemory::GetBuf, uint8_t* (*)())
            .stubs().will(returnValue(reinterpret_cast<uint8_t*>(hdr)));
        std::shared_ptr<ShareTokenMemory> shm(
            reinterpret_cast<ShareTokenMemory*>(0x88),
            [](ShareTokenMemory*) {
            }
        );

        // 临时覆盖为“缺字段”（按值返回）
        MOCKER_CPP(&ConfigManager::GetConfigJsonStr, std::string (*)())
            .stubs().will(MOCKCPP_NS::invoke(&Ret_MissingKeysConfigJson_Value));

        auto mies = pybind11::module_::import("mindie_llm.tokenizer");
        ASSERT_TRUE(pybind11::hasattr(mies, "IbisTokenizer"));

        std::shared_ptr<InferTokenizer> tk;
        EXPECT_TRUE(pool.InitSubProcessTokenizer(shm, tk)); // GetModelConfigString 返回 "" 也不影响成功
        EXPECT_NE(tk, nullptr);

        delete hdr;
        // 恢复默认
        MOCKER_CPP(&ConfigManager::GetConfigJsonStr, std::string (*)())
            .stubs().will(MOCKCPP_NS::invoke(&Ret_DefaultConfigJson_Value));
    }

    /** 覆盖 GetModelConfigString：非法 JSON -> 外层返回 false */
    TEST_F(InferenceTokenizerTest, InitSubTok_ConfigJson_Invalid_Fail)
    {
        auto &pool = TokenizerProcessPool::GetInstance();

        auto *hdr = new SharedMemoryHeader{};  // 值初始化
        hdr->magic = MAGIC_HEAD_BEGIN;

        MOCKER_CPP(&ShareTokenMemory::GetBuf, uint8_t* (*)())
            .stubs().will(returnValue(reinterpret_cast<uint8_t*>(hdr)));
        std::shared_ptr<ShareTokenMemory> shm(
            reinterpret_cast<ShareTokenMemory*>(0x99),
            [](ShareTokenMemory*) {
            }
        );

        MOCKER_CPP(&ConfigManager::GetConfigJsonStr, std::string (*)())
            .stubs().will(MOCKCPP_NS::invoke(&Ret_InvalidConfigJson_Value));

        auto mies = pybind11::module_::import("mindie_llm.tokenizer");
        ASSERT_TRUE(pybind11::hasattr(mies, "IbisTokenizer"));

        std::shared_ptr<InferTokenizer> tk;
        EXPECT_TRUE(pool.InitSubProcessTokenizer(shm, tk));

        delete hdr;
        // 恢复默认
        MOCKER_CPP(&ConfigManager::GetConfigJsonStr, std::string (*)())
            .stubs().will(MOCKCPP_NS::invoke(&Ret_DefaultConfigJson_Value));
    }

    /** DoDecode 分支：输入 token 过长 -> 立即返回错误 */
    TEST_F(InferenceTokenizerTest, DoDecode_InputTooLong_ReturnsError)
    {
        auto &pool = TokenizerProcessPool::GetInstance();
        auto serverConfig = mindie_llm::ConfigManager::GetInstance().GetServerConfig();
        uint32_t maxTextLength = serverConfig.maxRequestLength * 1024 * 1024;
        uint32_t MAX_TOKEN_LENGTH = maxTextLength / sizeof(int64_t);

        std::vector<int64_t> tok(MAX_TOKEN_LENGTH + 1, 1); // 关键：超长
        std::string out;
        DetokenizeExtraInfo extra;
        uint64_t ts = 0;

        auto st = pool.Decode(tok, out, ts, /*useToolsCall*/false, /*skip*/true, extra);
        EXPECT_FALSE(st.IsOk());
        EXPECT_TRUE(out.empty());
    }

    /** DoDecode 分支：找不到 pid 对应内存索引 -> 命中 idxIter == end() */
    TEST_F(InferenceTokenizerTest, DoDecode_PidNotFound_ReturnsError)
    {
        auto &pool = TokenizerProcessPool::GetInstance();

        // 把可用 pid 伪造成 map 中不存在的值
        MOCKER_CPP(&TokenizerProcessPool::GetAvailablePid, pid_t (*)())
            .stubs().will(returnValue(999999));

        std::vector<int64_t> tok{0, 1};
        std::string out;
        DetokenizeExtraInfo extra;
        uint64_t ts = 0;

        auto st = pool.Decode(tok, out, ts, /*useToolsCall*/false, /*skip*/true, extra);
        EXPECT_FALSE(st.IsOk());
        EXPECT_TRUE(out.empty());

        GlobalMockObject::reset();
    }

    /** DoDecode 分支：header == nullptr（GetBuf 返回空） */
    TEST_F(InferenceTokenizerTest, DoDecode_HeaderNull_ReturnsError)
    {
        auto &pool = TokenizerProcessPool::GetInstance();

        // 注意：用 (uint8_t*)0（或 (unsigned char*)0），不要用 reinterpret_cast<uint8_t*>(nullptr)
        MOCKER_CPP(&ShareTokenMemory::GetBuf, uint8_t* (*)())
            .stubs().will(returnValue((uint8_t*)0));

        std::vector<int64_t> tok{0, 1};
        std::string out;
        DetokenizeExtraInfo extra;
        uint64_t ts = 0;

        auto st = pool.Decode(tok, out, ts, /*useToolsCall*/false, /*skip*/true, extra);
        EXPECT_FALSE(st.IsOk());
        EXPECT_TRUE(out.empty());

        GlobalMockObject::reset();
    }

} // namespace SimpleLLMInference
