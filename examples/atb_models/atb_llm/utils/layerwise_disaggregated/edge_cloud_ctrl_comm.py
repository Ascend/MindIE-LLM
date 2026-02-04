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

CLOUD = 'slave'
EDGE = 'master'

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
    def __init__(self, server_ip, server_port, tls_config):
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
                self.client_socket.setblocking(False)
                logger.info(f"[layerwiseDisaggregated] Successfully connected to the TCP server \
                    {self.server_ip, self.server_port}.")
                return
            except Exception as e:
                logger.error(f"[layerwiseDisaggregated] Unable to connect to TCP server \
                    {self.server_ip, self.server_port}, the reason is {e}.")
                time.sleep(1)

    def client_send(self, data):
        try:
            self.client_socket.sendall(data.encode('utf-8'))
            logger.info(f"[layerwiseDisaggregated] TCP client successfully sent message to \
                {self.server_ip, self.server_port}, data is {data}.")
        except Exception as e:
            logger.error(f"[layerwiseDisaggregated] TCP client failed to sent message to \
                {self.server_ip, self.server_port}, the reason is {e}.")

    def client_recv(self):
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
    def __init__(self, host_ip, port, tls_config):
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
            self.server_socket.setblocking(False)
            self.clients.setblocking(False)

            self.clients.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            logger.info(f"[layerwiseDisaggregated] A client {client_address} is connected to the server.")
        except Exception as e:
            if self.running:
                logger.error(f"Failed to accept the TCP client connection, the reason is {e}")

    def server_send(self, data):
        try:
            self.clients.sendall(data.encode('utf-8'))
            logger.info(f"[layerwiseDisaggregated] TCP server successfully sent message to \
                {self.clients_addr, self.port}, data is {data}.")
        except Exception as e:
            logger.error(f"[layerwiseDisaggregated] TCP server failed to sent message to \
                {self.clients_addr, self.port}, the reason is {e}.")

    def server_recv(self):
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


class EdgeCloudCtrlComm():
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
        self.prefill_comm_finish_tcp = False
        self.prefill_comm_finish_irecv = False

        self.prefill_recv_msg = ''
        self.decode_recv_msg = ''
        self.prefill_send_msg = ''
        self.decode_send_msg = ''
        self.parse_msg_cnt = 0
        self.to_msg_cnt = 0
        
        self.tls_config = tls_config

    def init_role(self, role, server_ip, server_port):
        self.role = role

        self.server_ip = server_ip
        self.server_port = json.loads(server_port)

    def init_tcp_link(self, rank=None, role=None, server_ip=None, server_port=None):
        self.rank = rank
        if self.rank == 0:
            self.init_role(role, server_ip, server_port)

        if self.role == CLOUD and self.rank == 0:
            self.prefill_client = TCPClient(self.server_ip, int(self.server_port[0]), self.tls_config)
            self.decode_client = TCPClient(self.server_ip, int(self.server_port[1]), self.tls_config)
            self.prefill_client.connect_to_server_block()
            self.decode_client.connect_to_server_block()
            logger.info(f"[layerwiseDisaggregated] EdgeCloudCtrlComm port \
                {self.server_ip, self.server_port} init TCP client.")
        elif self.role == EDGE and self.rank == 0:
            self.prefill_server = TCPServer(self.server_ip, int(self.server_port[0]), self.tls_config)
            self.decode_server = TCPServer(self.server_ip, int(self.server_port[1]), self.tls_config)
            logger.info(f"[layerwiseDisaggregated] EdgeCloudCtrlComm port \
                {self.server_ip, self.server_port} init TCP server.")
        
        self.init_finish = True

    def is_edge_cloud_ctrl_comm_success(self):
        if self.rank != 0:
            return True

        if self.role == CLOUD:
            return self.prefill_client.is_client_connected() and self.decode_client.is_client_connected()
        else:
            return True

    def recv_prefill(self):
        if self.rank != 0:
            return

        if self.role == CLOUD:
            res = self.prefill_client.client_recv()
        else:
            res = self.prefill_server.server_recv()
        if res and res.startswith("pull"):
            self.prefill_recv_msg = res
            self.prefill_comm_finish_tcp = True

    def recv_decode(self):
        if self.rank != 0:
            return

        if self.role == CLOUD:
            res = self.decode_client.client_recv()
        else:
            res = self.decode_server.server_recv()
        if res and res.startswith("pull"):
            self.decode_recv_msg = res
            self.decode_comm_finish = True

    def send_prefill(self):
        if self.rank != 0:
            return

        if self.role == CLOUD:
            self.prefill_client.client_send(self.prefill_send_msg)
        else:
            self.prefill_server.server_send(self.prefill_send_msg)

    def send_decode(self):
        if self.rank != 0:
            return

        if self.role == CLOUD:
            self.decode_client.client_send(self.decode_send_msg)
        else:
            self.decode_server.server_send(self.decode_send_msg)

    def parse_shape(self, data):
        if not data.startswith("pull"):
            return []
        h_shape_d = list(json.loads(data.split('|')[1]))
        self.parse_msg_cnt += 1
        return h_shape_d

    def shape_to_msg(self, shape):
        if shape is None or len(shape) != 2:
            return None
        msg = f"pull|{json.dumps(list(shape))}|0"
        self.to_msg_cnt += 1
        return msg
