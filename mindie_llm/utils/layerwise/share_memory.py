#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import struct
import time
import mmap
import pickle
from typing import Optional
import posix_ipc

from mindie_llm.utils.log.logging import logger

SHARED_MEM_TYPE = 'i'


class SharedMemoryManager:
    _instance: Optional['SharedMemoryManager'] = None
    _shm_path = "/dev/shm/llm_share_memory"
    _sem_mutex_path = "/llm_semaphore"  
    _sem_date_ready_path = "/llm_data_ready_semaphore" 
    _shm_size = 4096
    _ptr = None
    _shm_fd = -1
    _sem = None
    _data_sem = None

    _is_producer = False
    _consumer_num = 0
    _share_mem_type = 'i'
    _mem_type_size = struct.calcsize(_share_mem_type)

    tmp_fd = None
    tmp_mmp = None
    tmp_sem = None
    tmp_mutex = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, parent_pid):
        tmp_str = '_' + str(parent_pid)
        self._shm_path += tmp_str
        self._sem_mutex_path += tmp_str
        self._sem_date_ready_path += tmp_str

    def __del__(self):
        if self._ptr:
            self._ptr.close()
        if self._shm_fd != -1:
            os.close(self._shm_fd)
        if self._sem:
            self._sem.close()
        if self._data_sem:
            self._data_sem.close()

    @staticmethod
    def dict_to_bytes(dict_data):
        try:
            return pickle.dumps(dict_data)
        except Exception as e:
            logger.error(f"[layerwiseDisaggregated] dict_to_bytes error is {e}")
            return b""

    @staticmethod
    def bytes_to_dict(byte_data):
        try:
            return pickle.loads(byte_data)
        except Exception as e:
            logger.error(f"[layerwiseDisaggregated] bytes_to_dict error is {e}")
            return {}

    # init interface
    def initialize(self, is_producer, consumer_num, share_mem_type='i'):
        self._initialize(is_producer, consumer_num, share_mem_type)
        if is_producer:
            self._initialize_mem_path()
        time.sleep(1)

        max_retries = 10000
        for _ in range(max_retries):
            try:
                self._shm_fd = os.open(self._shm_path, os.O_RDWR, mode=0o644)
                self._ptr = mmap.mmap(self._shm_fd, self._shm_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
                self._sem = posix_ipc.Semaphore(self._sem_mutex_path, posix_ipc.O_RDWR)
                self._data_sem = posix_ipc.Semaphore(self._sem_date_ready_path, posix_ipc.O_RDWR)
                logger.info(f"[layerwiseDisaggregated] successed initialize SharedMemoryManager, "
                            f"shm path: {self._shm_path}")
                return
            except Exception:
                time.sleep(0.1)
        raise TimeoutError(f"Failed to initialize SharedMemoryManager after {max_retries} retries")

    # 调用者保证信号量
    def notify_all_consumer_can_read(self):
        # write can_read
        can_read = [1] * self._consumer_num
        can_read_offset = 0
        for item in can_read:
            if isinstance(item, int):
                packed_item = struct.pack(self._share_mem_type, item)
            else:
                raise ValueError("Unsupported data type in can_read")

            self._ptr.seek(can_read_offset)
            can_read_offset += self._ptr.write(packed_item)

    # 调用者保证信号量
    def consumer_clear_can_read_flag(self, consumer_id):
        can_read_offset = self._mem_type_size * (consumer_id - 1)
        self._ptr.seek(can_read_offset)
        can_read = struct.pack(self._share_mem_type, 0)
        self._ptr.write(can_read)

    # 调用者保证信号量
    def consumer_check_is_can_read(self, consumer_id) -> bool:
        can_read_offset = self._mem_type_size * (consumer_id - 1)
        self._ptr.seek(can_read_offset)
        item = self._ptr.read(self._mem_type_size)
        can_read = struct.unpack(self._share_mem_type, item)[0]
        logger.info(
            f"[layerwiseDisaggregated] sem value is {self._data_sem.value} id: {consumer_id} can_read: {can_read}")
        if can_read == 0:
            return False
        else:
            return True

    # 共享内存暂时规划方式: consumer_num * size(can_read) + size(data) + data
    def write_list_memory(self, data_list: list) -> None:
        if not self._is_producer:
            logger.error("[layerwiseDisaggregated] is not producer, can't write mem")
            return

        for _ in range(10000):
            if self._data_sem.value != 0:
                time.sleep(0.0005)
            else:
                break
        logger.info(f"[layerwiseDisaggregated] write list: {data_list}")

        self._sem.acquire()
        try:
            # write data size
            data_size_offset = self._mem_type_size * self._consumer_num
            self._ptr.seek(data_size_offset)
            data_size = struct.pack(SHARED_MEM_TYPE, len(data_list))
            self._ptr.write(data_size)

            # write data
            data_offset = data_size_offset + self._mem_type_size
            for item in data_list:
                if isinstance(item, int):
                    packed_item = struct.pack(SHARED_MEM_TYPE, item)
                else:
                    raise ValueError("Unsupported data type in list")
                
                self._ptr.seek(data_offset)
                data_offset += self._ptr.write(packed_item)

            logger.info(f"[layerwiseDisaggregated] write complete: {data_list}")
            self._ptr.flush()

            # release sem
            for _ in range(self._consumer_num):
                self._data_sem.release()
        finally:
            self.notify_all_consumer_can_read()
            self._sem.release()

    # consumer_id is range from 1 to N, the producer id is 0.
    def read_list_memory(self, consumer_id, size: int = 1024):
        if consumer_id < 1 or consumer_id > self._consumer_num:
            logger.error(f"[layerwiseDisaggregated] consumer_id: {consumer_id} out of range:[1, {self._consumer_num}]")
            return None

        self._sem.acquire()
        if not self.consumer_check_is_can_read(consumer_id):
            self._sem.release()
            return None

        self._data_sem.acquire()
        logger.info(f"[layerwiseDisaggregated] now data sem value is {self._data_sem.value} id: {consumer_id}")

        data_size_offset = self._mem_type_size * (self._consumer_num)
        self._ptr.seek(data_size_offset)
        packed_len = self._ptr.read(self._mem_type_size)
        list_len = struct.unpack(SHARED_MEM_TYPE, packed_len)[0]

        data_list = []
        data_offset = data_size_offset + self._mem_type_size
        for _ in range(list_len):
            self._ptr.seek(data_offset)
            packed_item = self._ptr.read(self._mem_type_size)
            try:
                item = struct.unpack(SHARED_MEM_TYPE, packed_item)[0]
                data_list.append(item)
                data_offset += self._mem_type_size
                continue
            except struct.error as e:
                raise ValueError("Unknown data type in shared memory") from e
        logger.info(f"[layerwiseDisaggregated] read complete: {data_list}")

        self.consumer_clear_can_read_flag(consumer_id)
        self._sem.release()
        return data_list

    # 共享内存暂时规划方式: consumer_num * size(can_read) + size(data) + data
    def write_dict_memory(self, data_dict: dict) -> None:
        if not self._is_producer:
            logger.error("[layerwiseDisaggregated] is not producer, can't write mem")
            return

        logger.info(f"[layerwiseDisaggregated] write dict: {data_dict}")

        self._sem.acquire()
        try:
            # write data size
            data_size_offset = self._mem_type_size * self._consumer_num
            self._ptr.seek(data_size_offset)
            data_bytes = self.dict_to_bytes(data_dict)
            data_len = struct.pack(self._share_mem_type, len(data_bytes))
            self._ptr.write(data_len)

            # write data
            data_offset = data_size_offset + self._mem_type_size
            self._ptr.seek(data_offset)
            self._ptr.write(data_bytes)

            logger.info(f"[layerwiseDisaggregated] write complete: {data_dict}")
            self._ptr.flush()

            # release sem
            for _ in range(self._consumer_num):
                self._data_sem.release()
        finally:
            self.notify_all_consumer_can_read()
            self._sem.release()

    # consumer_id is range from 1 to N, the producer id is 0.
    def read_dict_memory(self, consumer_id) -> Optional[dict]:
        if consumer_id < 1 or consumer_id > self._consumer_num:
            logger.error(f"[layerwiseDisaggregated] consumer_id: {consumer_id} out of range:[1, {self._consumer_num}]")
            return None

        self._sem.acquire()
        if not self.consumer_check_is_can_read(consumer_id):
            self._sem.release()
            return None

        self._data_sem.acquire()
        logger.info(f"[layerwiseDisaggregated] now data sem value is {self._data_sem.value} id: {consumer_id}")

        data_size_offset = self._mem_type_size * (self._consumer_num)
        self._ptr.seek(data_size_offset)
        packed_len = self._ptr.read(self._mem_type_size)
        data_len = struct.unpack(self._share_mem_type, packed_len)[0]

        data_offset = data_size_offset + self._mem_type_size
        self._ptr.seek(data_offset)
        data_bytes = self._ptr.read(data_len)
        data_dict = self.bytes_to_dict(data_bytes)
        logger.info(f"[layerwiseDisaggregated] read complete: {data_dict}")

        self.consumer_clear_can_read_flag(consumer_id)
        self._sem.release()
        return data_dict

    def _initialize_mem_path(self):
        try:
            os.unlink(self._shm_path)
        except OSError:
            pass
        try:
            posix_ipc.unlink_semaphore(self._sem_mutex_path)
        except (posix_ipc.ExistentialError, OSError):
            pass
        try:
            posix_ipc.unlink_semaphore(self._sem_date_ready_path)
        except (posix_ipc.ExistentialError, OSError):
            pass
        self.tmp_fd = os.open(self._shm_path, os.O_CREAT | os.O_RDWR, mode=0o644)
        os.ftruncate(self.tmp_fd, self._shm_size)
        self.tmp_mmp = mmap.mmap(self.tmp_fd, self._shm_size, mmap.MAP_SHARED, mmap.PROT_WRITE)
        self.tmp_sem = posix_ipc.Semaphore(self._sem_date_ready_path, flags=posix_ipc.O_CREAT, initial_value=0)
        self.tmp_mutex = posix_ipc.Semaphore(self._sem_mutex_path, flags=posix_ipc.O_CREAT, initial_value=1)

    def _initialize(self, is_producer, consumer_num, share_mem_type='i'):
        if consumer_num < 1:
            raise ValueError("lwd share memory consumer_num can't be lower than 1.")
        self._is_producer = is_producer
        self._consumer_num = consumer_num
        self._share_mem_type = share_mem_type
        self._mem_type_size = struct.calcsize(share_mem_type)

