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
#include <chrono>
#include <thread>
#include <nlohmann/json.hpp>
#include <acl/acl.h>

#include <atb/types.h>
#include <atb/utils.h>
#include <atb/operation.h>

#include "atb_speed/utils/operation_util.h"
#include "atb_speed/utils/check_util.h"
#include "atb_speed/log.h"
#include "atb_speed/base/event_manager.h"

namespace atb_speed {

thread_local std::vector<std::pair<atb::Operation*, atb::common::EventParam>> g_eventOperationsOfModel;

#define CHECK_ACL_STATUS_RETURN_AND_LOG_IF_ERROR(ret, customErr, msg) \
    do { \
        if ((ret) != ACL_SUCCESS) { \
            ATB_SPEED_LOG_ERROR((msg) << " aclError = " << (ret)); \
            return (customErr); \
        } \
    } while (0)

#define CHECK_EM_STATUS_ONLY_RETURN(status, retVal) \
    do { \
        if ((status) != EM_SUCCESS) { \
            return (retVal); \
        } \
    } while (0)

#define CHECK_EM_STATUS_RETURN_AND_LOG_IF_ERROR(status, retVal, msg) \
    do { \
        if ((status) != EM_SUCCESS) { \
            ATB_SPEED_LOG_ERROR((msg) << " status=" << (status)); \
            return (retVal); \
        } \
    } while (0)

EventManager& EventManager::GetInstance()
{
    static EventManager instance;
    return instance;
}

EventManager::EventManager()
{
    ATB_SPEED_LOG_DEBUG("EventManager created.");
    uint32_t opWaitTimeout = 1080;
    SetWaitOperationTimeout(opWaitTimeout);
}

EventManager::~EventManager()
{
    std::lock_guard<std::mutex> lk(queueMutex_);
    for (auto& eventQueue : eventQueues_) {
        while (!eventQueue.second.empty()) {
            aclrtEvent event = eventQueue.second.front();
            eventQueue.second.pop();
            DestroyEvent(event);
        }
    }
    ATB_SPEED_LOG_DEBUG("EventManager destroyed.");
    for (auto& eventsTuple : eventsForExternal_) {
        auto& eventsForExtel = std::get<1>(eventsTuple.second);
        for (auto& eventE : eventsForExtel) {
            DestroyEvent(eventE);
        }
    }
    ATB_SPEED_LOG_DEBUG("EventManager destroyed.");
}

void EventManager::SetWaitOperationTimeout(uint32_t timeout)
{
    aclError ret = aclrtSetOpWaitTimeout(timeout);
    if (ret != ACL_SUCCESS) {
        ATB_SPEED_LOG_ERROR("aclrtSetOpWaitTimeout failed, aclError = " << ret);
    } else {
        ATB_SPEED_LOG_DEBUG("aclrtSetOpWaitTimeout end, set to " << timeout << " seconds.");
    }
}

EventManagerStatus EventManager::CreateEvent(aclrtEvent &event) const
{
    uint32_t flags = ACL_EVENT_SYNC;
    aclError ret = aclrtCreateEventWithFlag(&event, flags);
    CHECK_ACL_STATUS_RETURN_AND_LOG_IF_ERROR(ret, EM_CREATE_EVENT_FAILED, "aclrtCreateEventWithFlag failed,");
    ATB_SPEED_LOG_DEBUG("Event created, event = " << event);

    return EM_SUCCESS;
}

EventManagerStatus EventManager::DestroyEvent(aclrtEvent event) const
{
    aclError ret = aclrtDestroyEvent(event);
    CHECK_ACL_STATUS_RETURN_AND_LOG_IF_ERROR(ret, EM_DESTROY_EVENT_FAILED, "aclrtDestroyEvent failed,");
    ATB_SPEED_LOG_DEBUG("Event destroyed end, event = " << event);

    return EM_SUCCESS;
}

EventManagerStatus EventManager::CreateAndPushEvent(aclrtEvent &event, const std::string &pipeKey)
{
    if (eventQueues_.find(pipeKey) == eventQueues_.end()) {
        std::queue<aclrtEvent> queue;
        eventQueues_[pipeKey] = queue;
    }

    EventManagerStatus ret = CreateEvent(event);
    CHECK_EM_STATUS_ONLY_RETURN(ret, ret);
    std::lock_guard<std::mutex> lk(queueMutex_);
    eventQueues_[pipeKey].push(event);
    eventCount_.fetch_add(1, std::memory_order_relaxed);
    eventCond_.notify_one();

    ATB_SPEED_LOG_DEBUG("PushEvent: event = " << event
                        << ", Event pushed to queue, queueSize = " << eventQueues_[pipeKey].size()
                        << ", current eventCount = " << eventCount_.load());

    return EM_SUCCESS;
}

EventManagerStatus EventManager::PopEvent(aclrtEvent &event, const std::string &pipeKey)
{
    std::unique_lock<std::mutex> lk(queueMutex_);
    if (!eventCond_.wait_for(lk, std::chrono::microseconds(1),
        [this, pipeKey] { return !eventQueues_[pipeKey].empty(); })) {
        ATB_SPEED_LOG_DEBUG("PopEvent: Timeout waiting for event, current eventCount = " << eventCount_.load());
        return EM_POP_EVENT_TIMEOUT;
    }
    event = eventQueues_[pipeKey].front();
    eventQueues_[pipeKey].pop();
    eventCount_.fetch_sub(1, std::memory_order_relaxed);
    lk.unlock();

    ATB_SPEED_LOG_DEBUG("PopEvent: event = " << event
                        << ", current eventCount = " << eventCount_.load());

    return EM_SUCCESS;
}

atb::Status EventManager::EventInternal(EventAction eventAction,
                                        EventType eventType,
                                        atb::Operation*& op,
                                        const std::string &pipeKey)
{
    atb::common::EventParam eventParam;
    atb::common::EventParam::OperatorType opType;
    std::string eventTypeStr;

    if (eventType == EventType::RECORD) {
        opType = atb::common::EventParam::OperatorType::RECORD;
        eventTypeStr = "RecordEvent";
    } else if (eventType == EventType::WAIT) {
        opType = atb::common::EventParam::OperatorType::WAIT;
        eventTypeStr = "WaitEvent";
    } else {
        ATB_SPEED_LOG_ERROR("Invalid EventType: " << static_cast<int>(eventType));
        return EM_INVALID_TYPE;
    }
    eventParam.operatorType = opType;

    aclrtEvent event = nullptr;
    EventManagerStatus ret;

    if (eventAction == EventAction::PUSH) {
        ret = CreateAndPushEvent(event, pipeKey);
        CHECK_EM_STATUS_RETURN_AND_LOG_IF_ERROR(ret, EM_PUSH_EVENT_FAILED,
            eventTypeStr + ": CreateAndPushEvent failed with error");
        ATB_SPEED_LOG_DEBUG(eventTypeStr << ": Pushed event, event: " << event);
        if (!opsWithoutEvent_[pipeKey].empty()) {
            auto& opParam = opsWithoutEvent_[pipeKey].front();
            opsWithoutEvent_[pipeKey].pop();
            opParam.second.event = eventQueues_[pipeKey].front();
            eventQueues_[pipeKey].pop();
            atb::UpdateOperationParam(opParam.first, opParam.second);
            ATB_SPEED_LOG_DEBUG(eventTypeStr << ": Popped event, event: " << event);
        }
    } else if (eventAction == EventAction::POP) {
        ret = PopEvent(event, pipeKey);
        if (ret == EM_POP_EVENT_TIMEOUT) {
            ATB_SPEED_LOG_DEBUG(eventTypeStr << ": Popped event, event: time out");
            atb::common::EventParam eventParamTmp;
            if (atb::CreateOperation(eventParamTmp, &op) != atb::NO_ERROR) {
                return EM_OPERATION_CREATION_FAILED;
            }
            opsWithoutEvent_[pipeKey].push(std::make_pair(op, eventParam));
            return atb::NO_ERROR;
        }
        ATB_SPEED_LOG_DEBUG(eventTypeStr << ": Popped event, event: " << event);
    } else {
        return EM_INVALID_ACTION;
    }

    eventParam.event = event;

    if (atb::CreateOperation(eventParam, &op) != atb::NO_ERROR) {
        return EM_OPERATION_CREATION_FAILED;
    }

    g_eventOperationsOfModel.push_back(std::make_pair(op, eventParam));

    return atb::NO_ERROR;
}

atb::Status EventManager::RecordEvent(atb::Operation*& op, EventAction eventAction, const std::string &pipeKey)
{
    return EventInternal(eventAction, EventType::RECORD, op, pipeKey);
}

atb::Status EventManager::WaitEvent(atb::Operation*& op, EventAction eventAction, const std::string &pipeKey)
{
    return EventInternal(eventAction, EventType::WAIT, op, pipeKey);
}

EventManagerStatus EventManager::CheckPipeKey(const std::string &pipeKey)
{
    if (eventsForExternal_.find(pipeKey) == eventsForExternal_.end()) {
        if (eventQueues_.find(pipeKey) == eventQueues_.end()) {
            ATB_SPEED_LOG_DEBUG("Event for external error: no such pipekey" << pipeKey);
            return EM_INVALID_ACTION;
        }
        std::vector<aclrtEvent> queue;
        while (!eventQueues_[pipeKey].empty()) {
            queue.push_back(eventQueues_[pipeKey].front());
            eventQueues_[pipeKey].pop();
        }
        aclrtStream subStream;
        aclError ret = aclrtCreateStream(&subStream);
        if (ret != ACL_ERROR_NONE) {
            ATB_SPEED_LOG_ERROR("Failed to create aclrtStream: " << ret);
            return EM_INVALID_ACTION;
        }
        ret = aclrtSetStreamFailureMode(subStream, ACL_STOP_ON_FAILURE);
        if (ret != 0) {
            ATB_SPEED_LOG_ERROR("Failed to aclrtSetStreamFailureMode: " << ret);
            return EM_INVALID_ACTION;
        }
        eventsForExternal_[pipeKey] = std::make_tuple(0, queue, subStream);
    }
    return EM_SUCCESS;
}

EventManagerStatus EventManager::RecordEvent(const std::string &pipeKey)
{
    auto rt = CheckPipeKey(pipeKey);
    if (rt != EM_SUCCESS) {
        return rt;
    }
    auto& currentEventIdx = std::get<0>(eventsForExternal_[pipeKey]);
    auto& eventsForExtel = std::get<1>(eventsForExternal_[pipeKey]);
    auto& stream = std::get<2>(eventsForExternal_[pipeKey]);
    aclError ret = aclrtRecordEvent(eventsForExtel[currentEventIdx], stream);
    currentEventIdx = (currentEventIdx + 1) % eventsForExtel.size();
    if (ret != ACL_SUCCESS) {
        ATB_SPEED_LOG_ERROR("aclrtRecordEvent fail, ret: " << ret);
        return EM_INVALID_ACTION;
    }
    return EM_SUCCESS;
}

EventManagerStatus EventManager::WaitEvent(const std::string &pipeKey)
{
    auto rt = CheckPipeKey(pipeKey);
    if (rt != EM_SUCCESS) {
        return rt;
    }
    auto& currentEventIdx = std::get<0>(eventsForExternal_[pipeKey]);
    auto& eventsForExtel = std::get<1>(eventsForExternal_[pipeKey]);
    auto& stream = std::get<2>(eventsForExternal_[pipeKey]);
    aclError ret = aclrtStreamWaitEvent(stream, eventsForExtel[currentEventIdx]);
    if (ret != ACL_SUCCESS) {
        ATB_SPEED_LOG_ERROR("aclrtWaitEvent fail, ret: " << ret);
        return EM_INVALID_ACTION;
    }
    ret = aclrtResetEvent(eventsForExtel[currentEventIdx], stream);
    if (ret != ACL_SUCCESS) {
        ATB_SPEED_LOG_ERROR("aclrtResetEvent fail, ret: " << ret);
        return EM_INVALID_ACTION;
    }
    if (aclrtSynchronizeStream(stream) != 0) {
        ATB_SPEED_LOG_ERROR("EventManager::WaitEvent: aclrtSynchronizeStream fail");
        return EM_INVALID_ACTION;
    }
    currentEventIdx = (currentEventIdx + 1) % eventsForExtel.size();
    return EM_SUCCESS;
}
}  // namespace atb_speed