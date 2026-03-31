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

import json
import socket
import time
import ssl
import os
import sys
import ctypes
from ctypes import c_char_p
from pathlib import Path

from atb_llm.utils.log import logger
from atb_llm.utils.initial import NPUSocInfo

CLOUD = 'slave'
EDGE = 'master'
NPU_SMI_INFO_ACK = "npu_smi_info_ack"
NPU_SMI_INFO_NOTIFY = "npu_smi_info_notify"

LAYERWISE_DISAGGREGATED_TCP_BUFFER_SIZE = 1024 * 1024
INSTALL_PATH = "MINDIE_LLM_HOME_PATH"
TRUE = 'true'


class CertUtil:
    @classmethod
    def decrypt_password(cls, config: dict) -> str:
        libhse_cryption_so_path = os.path.join(os.getenv(INSTALL_PATH), "lib", "libhse_cryption.so")
        with open(config["tls_passwd"]) as f:
            cipher_text = f.read().strip()

        lib = ctypes.CDLL(libhse_cryption_so_path)
        lib.Decrypt.argtypes = [c_char_p, c_char_p, c_char_p, c_char_p]
        lib.Decrypt.restype = None

        plain_text = ctypes.create_string_buffer(33)
        lib.Decrypt(cipher_text.encode(), plain_text, config["kmc_ksf_master"].encode(),
                    config["kmc_ksf_standby"].encode())
        password = plain_text.value.decode()
        ctypes.memset(plain_text, 0, len(plain_text))
        del plain_text
        return password

    @classmethod
    def load_ca_certificates_from_dir(cls, ca_dir_path: str, context: ssl.SSLContext):
        ca_dir = Path(ca_dir_path)
        cert_files = []
        for ext in ('*.crt', '*.pem'):
            cert_files.extend(ca_dir.glob(ext))

        combined_cert_data = ""

        for cert_file in sorted(cert_files):
            try:
                with open(cert_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content.startswith("-----BEGIN CERTIFICATE-----"):
                        raise ValueError(f"not vaild PEM certificate: {cert_file}")
                    combined_cert_data += "\n" + content
            except Exception as e:
                raise ValueError(f"read certificate failed: {cert_file}, error: {e}") from e

        temp_cert_path = None
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as tmp:
                tmp.write(combined_cert_data)
                temp_cert_path = tmp.name

            context.load_verify_locations(cafile=temp_cert_path)

        except Exception as e:
            raise RuntimeError(f"combined cert failed: {e}") from e

        finally:
            if temp_cert_path and os.path.exists(temp_cert_path):
                os.unlink(temp_cert_path)


class TCPClient:
    def __init__(self, server_ip, server_port, tls_config, non_block=True):
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_socket = None
        self.connected = False
        self.recv_buf_size = LAYERWISE_DISAGGREGATED_TCP_BUFFER_SIZE
        
        self.tls_enable = True if tls_config.get("tls_enable", '0') == '1' else False
        if self.tls_enable:
            self.tls_ca_path = os.path.join(os.getenv(INSTALL_PATH), tls_config.get("tls_ca_path", ''))
            self.tls_cert = os.path.join(os.getenv(INSTALL_PATH), tls_config.get("tls_cert", ''))
            self.tls_pk = os.path.join(os.getenv(INSTALL_PATH), tls_config.get("tls_pk", ''))
            self.tls_crl_path = os.path.join(os.getenv(INSTALL_PATH), tls_config.get("tls_crl_path", ''))
            self.tls_crl_files = os.path.join(os.getenv(INSTALL_PATH), tls_config.get("tls_crl_files", ''))


        self.non_block = non_block

    def connect_to_server_block(self):
        # TCP attempts 1,000 connections; if connection fails after all 1,000 attempts, service startup fails.
        for _ in range(1000): 
            ssl_context = None
            if self.tls_enable:
                decrypt_config = {
                    "tls_passwd": self.tls_pk_pwd,
                    "kmc_ksf_master": self.kmc_ksf_master,
                    "kmc_ksf_standby": self.kmc_ksf_standby
                }
            try:
                if self.tls_enable:
                    password = CertUtil.decrypt_password(config=decrypt_config)
                    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS)
                    ssl_context.load_cert_chain(certfile=self.tls_cert, keyfile=self.tls_pk, password=password)
                    password_len = len(password)
                    password_offset = sys.getsizeof(password) - password_len - 1
                    ctypes.memset(id(password) + password_offset, 0, password_len)
                    del password
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_REQUIRED
                    CertUtil.load_ca_certificates_from_dir(self.tls_ca_path, ssl_context)
                    
                
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                if self.tls_enable:
                    self.client_socket = ssl_context.wrap_socket(self.client_socket, server_side=False)

                server_address = (self.server_ip, self.server_port)

                self.client_socket.connect(server_address)
                self.connected = True
                if self.non_block:
                    self.client_socket.setblocking(False)
                logger.info(f"[layerwiseDisaggregated] Successfully connected to the TCP server \
                    {self.server_ip, self.server_port}.")
                return
            except Exception as e:
                logger.warning(f"[layerwiseDisaggregated] Unable to connect to TCP server \
                    {self.server_ip, self.server_port}, the reason is {e}.")
                time.sleep(10)

    def send(self, data):
        try:
            self.client_socket.sendall(data.encode('utf-8'))
            logger.info(f"[layerwiseDisaggregated] TCP client successfully sent message to \
                {self.server_ip, self.server_port}, data is {data}.")
        except Exception as e:
            logger.error(f"[layerwiseDisaggregated] TCP client failed to sent message to \
                {self.server_ip, self.server_port}, the reason is {e}.")

    def recv(self):
        res = None
        try:
            res = self.client_socket.recv(self.recv_buf_size).decode('utf-8')
        except Exception as e:
            if not isinstance(e, BlockingIOError):
                logger.error(f"[layerwiseDisaggregated] TCP client failed to receive message from server \
                    {self.server_ip, self.server_port}, the reason is {e}.")
            return None
        return res

    def is_client_connected(self):
        return self.connected

    def disconnect(self):
        if self.client_socket:
            self.client_socket.close()
            self.client_socket = None
            self.connected = False
            logger.info("[layerwiseDisaggregated] TCP client disconnected from server.")


class TCPServer:
    def __init__(self, host_ip, port, tls_config, non_block=True):
        self.host_ip = host_ip
        self.port = port
        self.server_socket = None
        self.clients = None
        self.clients_addr = None
        self.running = False
        self.recv_buf_size = LAYERWISE_DISAGGREGATED_TCP_BUFFER_SIZE
        self.tls_enable = True if tls_config.get("tls_enable", '0') == '1' else False
        if self.tls_enable:
            self.tls_ca_path = os.path.join(os.getenv(INSTALL_PATH), tls_config.get("tls_ca_path", ''))
            self.tls_cert = os.path.join(os.getenv(INSTALL_PATH), tls_config.get("tls_cert", ''))
            self.tls_pk = os.path.join(os.getenv(INSTALL_PATH), tls_config.get("tls_pk", ''))
            self.tls_crl_path = os.path.join(os.getenv(INSTALL_PATH), tls_config.get("tls_crl_path", ''))
            self.tls_crl_files = os.path.join(os.getenv(INSTALL_PATH), tls_config.get("tls_crl_files", ''))


        self.non_block = non_block
        self.start_server_block()

    def start_server_block(self):
        ssl_context = None
        if self.tls_enable:
            decrypt_config = {
                "tls_passwd": self.tls_pk_pwd,
                "kmc_ksf_master": self.kmc_ksf_master,
                "kmc_ksf_standby": self.kmc_ksf_standby
            }
        try:
            if self.tls_enable:
                password = CertUtil.decrypt_password(config=decrypt_config)
                ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                ssl_context.load_cert_chain(certfile=self.tls_cert, keyfile=self.tls_pk, password=password)
                password_len = len(password)
                password_offset = sys.getsizeof(password) - password_len - 1
                ctypes.memset(id(password) + password_offset, 0, password_len)
                del password

                CertUtil.load_ca_certificates_from_dir(self.tls_ca_path, ssl_context)
                ssl_context.verify_mode = ssl.CERT_REQUIRED
                ssl_context.check_hostname = False

            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.server_socket.bind((self.host_ip, self.port))

            if self.tls_enable:
                self.server_socket = ssl_context.wrap_socket(self.server_socket, server_side=True) 

            self.server_socket.listen(1)
            self.running = True
            logger.info(f"[layerwiseDisaggregated] TCP server starts listening address {self.host_ip}:{self.port}.")
            client_socket, client_address = self.server_socket.accept()
            self.clients = client_socket
            self.clients_addr = client_address
            if self.non_block:
                self.server_socket.setblocking(False)
                self.clients.setblocking(False)

            self.clients.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            logger.info(f"[layerwiseDisaggregated] A client {client_address} is connected to the server.")
        except Exception as e:
            if self.running:
                logger.error(f"Failed to accept the TCP client connection, the reason is {e}")

    def send(self, data):
        try:
            self.clients.sendall(data.encode('utf-8'))
            logger.info(f"[layerwiseDisaggregated] TCP server successfully sent message to \
                {self.clients_addr, self.port}, data is {data}.")
        except Exception as e:
            logger.error(f"[layerwiseDisaggregated] TCP server failed to sent message to \
                {self.clients_addr, self.port}, the reason is {e}.")

    def recv(self):
        res = None
        try:
            res = self.clients.recv(self.recv_buf_size).decode('utf-8')
        except Exception as e:
            if not isinstance(e, BlockingIOError):
                logger.error(f"[layerwiseDisaggregated] TCP server failed to receive message from client \
                    {self.clients_addr, self.port}, the reason is {e}.")
            return None
        return res

    def close_server(self):
        try:
            if self.clients:
                self.clients.close()
            self.server_socket.close()
        except Exception as e:
            logger.info(f"[layerwiseDisaggregated] TCP client {self.clients} \
                disconnected from server error, the reason is {e}.")
        finally:
            self.server_socket.close()


class EdgeCloudCtrlComm:
    def __init__(self, tls_config):
        self.role = None
        self.rank = None
        self.server_ip = ''
        self.server_port = ''
        self.init_finish = False

        self.prefill_server = None
        self.decode_server = None
        self.prefill_client = None
        self.decode_client = None

        self.decode_comm_finish = False
        self.prefill_comm_finish = False
        self.prefill_comm_finish_tcp_count = 0

        self.prefill_recv_msg = ''
        self.decode_recv_msg = ''
        self.prefill_send_msg = ''
        self.decode_send_msg = ''
        
        self.multi_nodes_infer_enabled = False
        self.multi_nodes_is_master = False
        self.multi_nodes_ctrl_server = None
        self.multi_nodes_ctrl_client = None
        self.multi_nodes_need_synchronize = False

        self.comm_group_size = 1

        self.tls_config = tls_config
        self.edge_npu_smi_info = None
        self.cloud_npu_smi_info = None

    @staticmethod
    def shape_to_msg(shape):
        if shape is None or len(shape) != 2:
            return None
        msg = f"pull|{json.dumps(list(shape))}|0"
        return msg

    def init_role(self, role, server_ip, server_port):
        self.role = role

        self.server_ip = server_ip
        self.server_port = json.loads(server_port)

    def init_multi_nodes_infer_link(self, multi_nodes_config):
        self.multi_nodes_infer_enabled = True
        self.comm_group_size = multi_nodes_config.get('comm_group_size')

        is_master = multi_nodes_config.get('is_master', False)
        cloud_ctrl_address = multi_nodes_config.get('cloud_ctrl_address')
        cloud_ctrl_port = multi_nodes_config.get('cloud_ctrl_port')
        
        self.multi_nodes_is_master = is_master
        if self.multi_nodes_is_master:
            self.multi_nodes_ctrl_server = TCPServer(cloud_ctrl_address, int(cloud_ctrl_port), \
                    self.tls_config, non_block=False)
            logger.info(f"[layerwiseDisaggregated] EdgeCloudCtrlComm multi-nodes infer \
                    {cloud_ctrl_address, cloud_ctrl_port} init TCP server.")
        else:
            self.multi_nodes_ctrl_client = TCPClient(cloud_ctrl_address, int(cloud_ctrl_port), \
                    self.tls_config, non_block=False)
            self.multi_nodes_ctrl_client.connect_to_server_block()
            logger.info(f"[layerwiseDisaggregated] EdgeCloudCtrlComm multi-nodes infer \
                    {cloud_ctrl_address, cloud_ctrl_port} init TCP client.")

    def init_tcp_link(self, rank=None, role=None, server_ip=None, server_port=None,
                        multi_nodes_infer_args=None):
        self.rank = rank
        self.init_role(role, server_ip, server_port)
        
        is_cloud_or_single_node = \
            (self.role == CLOUD or multi_nodes_infer_args is None or multi_nodes_infer_args['comm_group_size'] < 2)
        if self.rank != 0 and is_cloud_or_single_node:
            self.init_finish = True
            return

        if self.role == CLOUD:
            if multi_nodes_infer_args is not None:
                self.init_multi_nodes_infer_link(multi_nodes_infer_args)
            if multi_nodes_infer_args is None or self.comm_group_size > 1 or multi_nodes_infer_args['is_master']:
                self.prefill_server = TCPServer(self.server_ip, int(self.server_port[0]), self.tls_config)
                self.decode_server = TCPServer(self.server_ip, int(self.server_port[1]), self.tls_config)
                logger.info(f"[layerwiseDisaggregated] EdgeCloudCtrlComm \
                        {self.server_ip, self.server_port} TCP server initialized successfully.")
        elif self.role == EDGE:
            self.prefill_client = TCPClient(self.server_ip, int(self.server_port[0]), self.tls_config)
            self.prefill_client.connect_to_server_block()
            self.decode_client = TCPClient(self.server_ip, int(self.server_port[1]), self.tls_config)
            self.decode_client.connect_to_server_block()
            logger.info(f"[layerwiseDisaggregated] EdgeCloudCtrlComm \
                {self.server_ip, self.server_port} TCP client initialized successfully.")
        else:
            raise RuntimeError(f"EdgeCloudCtrlComm unknown role: {self.role}")
        
        self.init_finish = True

    def is_edge_cloud_ctrl_comm_success(self):
        prefill_clinet_comm_success = self.prefill_client is None or self.prefill_client.is_client_connected()
        decode_clinet_comm_success = self.decode_client is None or self.decode_client.is_client_connected()
        multi_nodes_comm_success = self.multi_nodes_ctrl_client is None or \
            self.multi_nodes_ctrl_client.is_client_connected()
        return prefill_clinet_comm_success and decode_clinet_comm_success and multi_nodes_comm_success

    def recv_prefill(self):
        res = None

        if self.role == CLOUD and self.prefill_server is not None:
            res = self.prefill_server.recv()
        if self.role == EDGE and self.prefill_client is not None:
            res = self.prefill_client.recv()

        if res and res.startswith("pull"):
            logger.info(f"[layerwiseDisaggregated-{self.rank}] recv valid prefill msg: {res}.")
            self.prefill_recv_msg = res
            self.prefill_comm_finish_tcp_count = res.count("pull")
        else:
            logger.info(f"[layerwiseDisaggregated-{self.rank}] recv no valid prefill msg!")

    def recv_decode(self):
        res = None

        if self.role == CLOUD and self.decode_server is not None:
            res = self.decode_server.recv()
        if self.role == EDGE and self.decode_client is not None:
            res = self.decode_client.recv()

        if res and res.startswith("pull"):
            logger.info(f"[layerwiseDisaggregated-{self.rank}] recv valid decode msg: {res}.")
            self.decode_recv_msg = res
            self.decode_comm_finish = True
        else:
            logger.info(f"[layerwiseDisaggregated-{self.rank}] recv no valid decode msg!")

    def send_prefill(self):
        if self.role == CLOUD and self.prefill_server is not None:
            self.prefill_server.send(self.prefill_send_msg)
            logger.info(f"[layerwiseDisaggregated-{self.rank}] send prefill msg: {self.prefill_send_msg}!")
            return

        if self.role == EDGE and self.prefill_client is not None:
            self.prefill_client.send(self.prefill_send_msg)
            logger.info(f"[layerwiseDisaggregated-{self.rank}] send prefill msg: {self.prefill_send_msg}!")
            return

        logger.info(f"[layerwiseDisaggregated-{self.rank}] skip send prefill msg!")

    def send_decode(self):
        if self.role == CLOUD and self.decode_server is not None:
            self.decode_server.send(self.decode_send_msg)
            logger.info(f"[layerwiseDisaggregated-{self.rank}] send decode msg: {self.decode_send_msg}!")
            return

        if self.role == EDGE and self.decode_client is not None:
            self.decode_client.send(self.decode_send_msg)
            logger.info(f"[layerwiseDisaggregated-{self.rank}] send decode msg: {self.decode_send_msg}!")
            return

        logger.info(f"[layerwiseDisaggregated-{self.rank}] skip send decode msg!")

    def broadcast_multi_nodes_decision(self, decision: str) -> None:
        if self.rank != 0 or not self.multi_nodes_infer_enabled:
            logger.info(f"[layerwiseDisaggregated-{self.rank}] skip broadcast multi nodes decision!")
            return
        if not self.multi_nodes_is_master:
            raise RuntimeError("EdgeCloudCtrlComm only cloud master can broadcast decision")
        if self.multi_nodes_need_synchronize:
            _ = self.multi_nodes_ctrl_server.recv()
        else:
            self.multi_nodes_need_synchronize = True

        self.multi_nodes_ctrl_server.send(decision)
        logger.info(f"[layerwiseDisaggregated-{self.rank}] broadcast multi nodes decision {decision}.")

    def recv_multi_nodes_decision(self) -> str | None:
        if self.rank != 0 or not self.multi_nodes_infer_enabled:
            logger.info(f"[layerwiseDisaggregated-{self.rank}] skip recv multi nodes decision!")
            return None
        if self.multi_nodes_is_master:
            raise RuntimeError("EdgeCloudCtrlComm only cloud slave can broadcast decision")

        decision = self.multi_nodes_ctrl_client.recv()
        self.multi_nodes_ctrl_client.send("ok")
        logger.info(f"[layerwiseDisaggregated-{self.rank}] recv multi nodes decision {decision}.")
        return decision

    def wait_recv(self, ctrl_comm_socket, wait_str):
        for _ in range(1000):
            res = ctrl_comm_socket.recv()
            if res and res.startswith(wait_str):
                return res
            logger.info(f"[layerwiseDisaggregated-{self.rank}] control communication recv {res}.")
            # 防止TCP粘包，时间间隔小一些
            time.sleep(0.5)
        logger.error(f"[layerwiseDisaggregated-{self.rank}] control communication recv {wait_str} timeout.")
        return None

    def npu_smi_info_sync_is_done(self):
        if self.role == EDGE:
            if self.rank != 0:
                return True if self.edge_npu_smi_info is not None else False
            else:
                return True if self.edge_npu_smi_info is not None and self.cloud_npu_smi_info is not None else False
        else:
            if self.rank != 0:
                return True if self.cloud_npu_smi_info is not None else False
            else:
                return True if self.edge_npu_smi_info is not None and self.cloud_npu_smi_info is not None else False

    def npu_smi_info_sync_edge_to_cloud(self, npu_smi_info: dict):
        if self.role == EDGE:
            npu_smi_info.update({'communication_backend': NPUSocInfo().communication_backend.name})
            logger.info(f"[layerwiseDisaggregated-{self.rank}] edge_npu_smi_info {npu_smi_info}.")

            if self.prefill_client is not None:
                msg = f"{NPU_SMI_INFO_NOTIFY}|{json.dumps(npu_smi_info)}"
                self.prefill_client.send(msg)
                logger.info(f"[layerwiseDisaggregated-{self.rank}] edge send edge_npu_smi_info notify to cloud.")
                if self.wait_recv(self.prefill_client, NPU_SMI_INFO_ACK) is not None:
                    logger.info(f"[layerwiseDisaggregated-{self.rank}] edge recv edge_npu_smi_info ack from cloud.")
                    self.edge_npu_smi_info = npu_smi_info
                else:
                    logger.error(f"[layerwiseDisaggregated-{self.rank}] edge recv edge_npu_smi_info ack timeout.")
            else:
                self.edge_npu_smi_info = npu_smi_info
        else:
            if self.rank != 0:
                return
            if self.prefill_server is not None:
                res = self.wait_recv(self.prefill_server, NPU_SMI_INFO_NOTIFY)
                if res is None:
                    logger.error(f"[layerwiseDisaggregated-{self.rank}] cloud recv edge_npu_smi_info notify timeout.")
                    return
                logger.info(f"[layerwiseDisaggregated-{self.rank}] cloud recv edge_npu_smi_info notify from edge.")
                npu_smi_info_str = res.lstrip(NPU_SMI_INFO_NOTIFY + "|")

                if self.multi_nodes_ctrl_server is not None and self.comm_group_size < 2:
                    self.multi_nodes_ctrl_server.send(res)
                    logger.info(f"[layerwiseDisaggregated-{self.rank}] cloud master "
                                "send edge_npu_smi_info notify to cloud slave.")
                    if self.wait_recv(self.multi_nodes_ctrl_server, NPU_SMI_INFO_ACK) is None:
                        logger.error(f"[layerwiseDisaggregated-{self.rank}] cloud master "
                                    "recv edge_npu_smi_info ack from cloud slave timeout.")
                        return
                    logger.info(f"[layerwiseDisaggregated-{self.rank}] cloud master "
                                "recv edge_npu_smi_info ack from cloud slave.")

                self.prefill_server.send(NPU_SMI_INFO_ACK)
                logger.info(f"[layerwiseDisaggregated-{self.rank}] cloud send edge_npu_smi_info ack to edge.")
                self.edge_npu_smi_info = json.loads(npu_smi_info_str)
            else:
                if self.multi_nodes_ctrl_client is not None:
                    res = self.wait_recv(self.multi_nodes_ctrl_client, NPU_SMI_INFO_NOTIFY)
                    if res is None:
                        logger.error(f"[layerwiseDisaggregated-{self.rank}] cloud slave "
                                    "recv edge_npu_smi_info notify from cloud master timeout.")
                        return
                    logger.info(f"[layerwiseDisaggregated-{self.rank}] cloud slave "
                                "recv edge_npu_smi_info notify from cloud master.")
                    npu_smi_info_str = res.lstrip(NPU_SMI_INFO_NOTIFY + "|")

                    self.multi_nodes_ctrl_client.send(NPU_SMI_INFO_ACK)
                    logger.info(f"[layerwiseDisaggregated-{self.rank}] cloud slave "
                                "send edge_npu_smi_info ack to cloud master.")
                    self.edge_npu_smi_info = json.loads(npu_smi_info_str)

    def npu_smi_info_sync_cloud_to_edge(self, npu_smi_info: dict):
        if self.role == CLOUD:
            npu_smi_info.update({'communication_backend': NPUSocInfo().communication_backend.name})
            logger.info(f"[layerwiseDisaggregated-{self.rank}] cloud_npu_smi_info {npu_smi_info}.")

            # 云侧双机认为npuinfo完全一样，只需要双机主向边侧发送
            if self.prefill_server is not None and (not self.multi_nodes_infer_enabled or self.multi_nodes_is_master):
                msg = f"{NPU_SMI_INFO_NOTIFY}|{json.dumps(npu_smi_info)}"
                self.prefill_server.send(msg)
                logger.info(f"[layerwiseDisaggregated-{self.rank}] cloud send cloud_npu_smi_info notify to edge.")
                if self.wait_recv(self.prefill_server, NPU_SMI_INFO_ACK) is not None:
                    logger.info(f"[layerwiseDisaggregated-{self.rank}] cloud recv cloud_npu_smi_info ack to edge.")
                    self.cloud_npu_smi_info = npu_smi_info
                else:
                    logger.error(f"[layerwiseDisaggregated-{self.rank}] cloud recv cloud_npu_smi_info ack timeout.")
            else:
                self.cloud_npu_smi_info = npu_smi_info
        else:
            if self.rank == 0 and self.prefill_client is not None:
                res = self.wait_recv(self.prefill_client, NPU_SMI_INFO_NOTIFY)
                if res is None:
                    logger.error(f"[layerwiseDisaggregated-{self.rank}] edge recv cloud_npu_smi_info notify timeout.")
                    return
                logger.info(f"[layerwiseDisaggregated-{self.rank}] edge recv cloud_npu_smi_info notify from cloud.")
                npu_smi_info_str = res.lstrip(NPU_SMI_INFO_NOTIFY + "|")

                self.prefill_client.send(NPU_SMI_INFO_ACK)
                logger.info(f"[layerwiseDisaggregated-{self.rank}] edge send cloud_npu_smi_info ack to cloud.")
                self.cloud_npu_smi_info = json.loads(npu_smi_info_str)

    def npu_smi_info_sync(self, npu_smi_info: dict):
        self.npu_smi_info_sync_edge_to_cloud(npu_smi_info)
        # 防止TCP粘包，设置2秒时间间隔
        time.sleep(2)
        self.npu_smi_info_sync_cloud_to_edge(npu_smi_info)