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

#ifndef ATB_SPEED_EVENT_MANAGER_H
#define ATB_SPEED_EVENT_MANAGER_H

#include <atb/context.h>
#include <atb/operation.h>
#include <atb/atb_infer.h>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <map>
#include <thread>
#include <atomic>
#include <nlohmann/json.hpp>

#include "atb_speed/log.h"

namespace atb_speed {
extern thread_local std::vector<std::pair<atb::Operation*, atb::common::EventParam>> g_eventOperationsOfModel;

// 定义EventManager返回状态码：
enum EventManagerStatus {
    EM_SUCCESS = 0, // 操作成功
    EM_CREATE_EVENT_FAILED = 1, // 使用 ACL 接口创建事件失败
    EM_PUSH_EVENT_FAILED = 2, // 推入事件队列失败
    EM_POP_EVENT_FAILED = 3, // 取出事件失败
    EM_POP_EVENT_TIMEOUT = 4, // 从事件队列中等待事件超时
    EM_INVALID_ACTION = 5, // 传入了无效的事件操作（既不是 PUSH 也不是 POP）
    EM_INVALID_TYPE = 6, // 传入了无效的事件类型（既不是 RECORD 也不是 WAIT）
    EM_DESTROY_EVENT_FAILED = 7, // 使用 ACL 接口销毁事件失败
    EM_OPERATION_CREATION_FAILED = 8, // 创建记录/等待操作节点失败（调用 atb::CreateOperation 失败）
    EM_INVALID_KEY = 9, // 传入了不存在或不合法的事件映射键
    EM_UNKNOWN_ERROR = 10, // 未知错误，预留状态码
};

// 定义事件类型：
enum class EventType {
    UNDEFINED, // 不执行事件操作
    RECORD,    // 记录事件操作
    WAIT,      // 等待事件操作
};

// 定义事件动作：
enum class EventAction {
    PUSH, // 创建新事件并入队
    POP,  // 从队列中取出已有事件
};

class EventManager {
public:
    // 单例访问接口，返回唯一的 EventManager 实例
    static EventManager& GetInstance();

    EventManager(const EventManager&) = delete;
    EventManager& operator=(const EventManager&) = delete;

    /**
     * @brief 设置 ACL 的操作等待超时时间
     * @param timeout 超时时间（单位：秒）
     */
    void SetWaitOperationTimeout(uint32_t timeout);

    //======================================================
    // 【多流接口】
    // - RecordEvent: 在多流场景下，通过 PUSH 操作创建记录事件并将其推入队列，或通过 POP 操作从事件队列中获取事件。
    // - WaitEvent: 在多流场景下，通过 PUSH 操作创建等待事件并将其推入队列，或通过 POP 操作从事件队列中获取事件。
    //------------------------------------------------------

    /**
     * @brief 在多流场景下，执行记录事件操作，该方法会根据 eventAction（PUSH/POP）执行相应的事件操作。
     * @param op 传出参数，创建的操作节点
     * @param eventAction 事件操作类型（PUSH 或 POP）
     * @param pipeKey 事件管道的标识符（默认为 "default"）
     * @return 返回操作的状态码（atb::Status）
     */
    atb::Status RecordEvent(atb::Operation*& op, EventAction eventAction, const std::string &pipeKey = "default");

    /**
     * @brief 在多流场景下，执行等待事件操作，该方法会根据 eventAction（PUSH/POP）执行相应的事件操作。
     * @param op 传出参数，创建的操作节点
     * @param eventAction 事件操作类型（PUSH 或 POP）
     * @param pipeKey 事件管道的标识符（默认为 "default"）
     * @return 返回操作的状态码（atb::Status）
     */
    atb::Status WaitEvent(atb::Operation*& op, EventAction eventAction, const std::string &pipeKey = "default");

    /**
     * @brief 在多流场景下，用于图外部与图内部的流间同步，取pipe中的current event record一次，current event指向下一个event，循环event队列。
     在同一张图里RecordEvent和WaitEvent只能使用同一个pipekey，否则会有时序问题。
     * @param pipeKey 事件管道的标识符（默认为 "default"）
     * @return 返回操作的状态码（atb::Status）
     */
    EventManagerStatus RecordEvent(const std::string &pipeKey = "default");

    /**
     * @brief 在多流场景下，用于图外部与图内部的流间同步，取pipe中的current event wait一次然后sync一次，current event指向下一个event，循环event队列。
     在同一张图里RecordEvent和WaitEvent只能使用同一个pipekey，否则会有时序问题。
     * @param pipeKey 事件管道的标识符（默认为 "default"）
     * @return 返回操作的状态码（atb::Status）
     */
    EventManagerStatus WaitEvent(const std::string &pipeKey = "default");

private:
    // 构造和析构函数设为 private，确保单例模式
    // 设置 ACL 的操作等待超时时间，默认 180 秒
    EventManager();
    ~EventManager();

    /**
     * @brief 使用 ACL 接口创建事件（默认为 ACL_EVENT_SYNC 类型）
     * @param event 通过引用返回创建的事件
     * @return EM_SUCCESS 或 EM_CREATE_EVENT_FAILED
     */
    EventManagerStatus CreateEvent(aclrtEvent &event) const;

    /**
     * @brief 销毁事件
     * @param event 待销毁的事件
     * @return EM_SUCCESS 或 EM_DESTROY_EVENT_FAILED
     */
    EventManagerStatus DestroyEvent(aclrtEvent event) const;

    /**
     * @brief 创建事件并将其推入队列
     * @param event 通过引用返回创建的新 event，并将其入队
     * @param pipeKey 用于区分不同事件管道的键
     * @return EM_SUCCESS 或相应的错误码
     */
    EventManagerStatus CreateAndPushEvent(aclrtEvent &event, const std::string &pipeKey);

    /**
     * @brief 从事件队列中取出一个事件
     * @param event 通过引用返回取出的事件
     * @param pipeKey 用于区分不同事件管道的键
     * @return EM_SUCCESS 或 EM_POP_EVENT_TIMEOUT
     */
    EventManagerStatus PopEvent(aclrtEvent &event, const std::string &pipeKey);

    /**
     * @brief 内部通用方法，用于处理记录/等待事件的公共逻辑
     * 该方法会根据 eventAction（PUSH/POP）和 eventType（RECORD/WAIT）获取 ACL 事件，并构造好事件参数（eventParam）。
     * @param eventAction 事件动作（PUSH 或 POP）
     * @param eventType 事件类型（RECORD 或 WAIT）
     * @param op 传出参数，创建的操作节点
     * @param pipeKey 事件管道的标识符
     * @return 返回操作的状态码（atb::Status）
     */
    atb::Status EventInternal(EventAction eventAction,
                              EventType eventType,
                              atb::Operation*& op,
                              const std::string &pipeKey);
    
    EventManagerStatus CheckPipeKey(const std::string &pipeKey);

private:
    // 条件变量，用于在事件队列为空时阻塞等待，并在有新事件入队时通知等待线程
    std::condition_variable eventCond_;
    // 事件队列：仅用于对外提供 push/pop 接口
    std::map<std::string, std::queue<aclrtEvent>> eventQueues_;
    std::map<std::string, std::queue<std::pair<atb::Operation*, atb::common::EventParam>>> opsWithoutEvent_;
    // 用于图外部与图内部的流间同步
    std::map<std::string, std::tuple<int, std::vector<aclrtEvent>, aclrtStream>> eventsForExternal_;
    // 保护 eventQueue_ 的互斥锁
    std::mutex queueMutex_;
    // 记录当前队列中 event 的数量
    std::atomic<uint64_t> eventCount_{0};
};
}  // namespace atb_speed

#endif  // ATB_SPEED_EVENT_MANAGER_H