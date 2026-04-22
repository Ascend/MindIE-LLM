/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include <mockcpp/mockcpp.hpp>

#include "pid_manage.h"

using namespace mindie_llm;

class PidManagerTest : public testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(PidManagerTest, AddIgnorePid) {
    pid_t pid = 0;
    pid_t pid1 = 1;
    EXPECT_EQ(false, PidManager::Instance().IsIgnorePid(pid));
    EXPECT_EQ(false, PidManager::Instance().IsIgnorePid(pid1));

    PidManager::Instance().AddIgnorePid(pid);
    EXPECT_EQ(true, PidManager::Instance().IsIgnorePid(pid));
    EXPECT_EQ(false, PidManager::Instance().IsIgnorePid(pid1));
}
