# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import unittest
import gc
import json
import ctypes
import os
import tempfile
import ssl

from unittest.mock import patch, MagicMock, call, mock_open
from atb_llm.utils.layerwise_disaggregated.edge_cloud_ctrl_comm import (
    CertUtil, EdgeCloudCtrlComm, TCPServer, TCPClient, LAYERWISE_DISAGGREGATED_TCP_BUFFER_SIZE
)
CLOUD = 'slave'
EDGE = 'master'


class TestCertUtil(unittest.TestCase):

    def setUp(self):
        self.original_mies_install_path = os.getenv("MINDIE_LLM_HOME_PATH")
        os.environ["MINDIE_LLM_HOME_PATH"] = "/fake/mies/install"
        self.temp_dir = tempfile.mkdtemp()
        self.test_cert_file = os.path.join(self.temp_dir, "test.crt")
        self.test_pass_file = os.path.join(self.temp_dir, "passwd.txt")

        with open(self.test_cert_file, 'w', encoding='utf-8') as f:
            f.write("-----BEGIN CERTIFICATE-----\n"
                     "MIIBxTCCAU+gAwIBAgIJAIP7XW917WUwDQYJKoZIhvcNAQEL\n"
                     "BQAwgYsxCzAJBgNVBAYTAkNBMQ8wDQYDVQQIDAZXaW5kcm93\n"
                     "MB4XDTIzMDExNDE0MjAwN1oXDTI0MDExNDE0MjAwN1owgYsxCzAJ\n"
                     "BgNVBAYTAkNBMQ8wDQYDVQQIDAZXaW5kcm93\n"
                     "-----END CERTIFICATE-----\n")

        with open(self.test_pass_file, 'w', encoding='utf-8') as f:
            f.write("encrypted_pass_123456")

    def tearDown(self):
        if self.original_mies_install_path is None:
            os.environ.pop("MINDIE_LLM_HOME_PATH", None)
        else:
            os.environ["MINDIE_LLM_HOME_PATH"] = self.original_mies_install_path

        if os.path.exists(self.test_cert_file):
            os.remove(self.test_cert_file)
        if os.path.exists(self.test_pass_file):
            os.remove(self.test_pass_file)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    @patch('ctypes.create_string_buffer')
    @patch('os.path.join')
    @patch('os.getenv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('ctypes.CDLL')
    def test_decrypt_password_success(self, mock_cdll, mock_open_fun, mock_getenv, mock_path_join, mock_buffer):
        mock_getenv.return_value = "/fake/mies/install"
        mock_path_join.return_value = "/fake/mies/install/lib/libhse_cryption.so"
        mock_open_fun.return_value.__enter__.return_value.read.return_value = "fake_cipher_text"
        mock_lib = MagicMock()
        mock_cdll.return_value = mock_lib
        
        patch('ctypes.memset', MagicMock())

        buftype = ctypes.c_char * 33
        buffer = buftype()
        buffer.value = b"mysecretpassword"
        mock_buffer.return_value = buffer

        config = {
            "tls_passwd": self.test_pass_file,
            "kmc_ksf_master": "master_key",
            "kmc_ksf_standby": "standby_key"
        }

        result = CertUtil.decrypt_password(config)
        self.assertEqual(result, "mysecretpassword")

        mock_path_join.assert_called_once_with(
            "/fake/mies/install", "lib", "libhse_cryption.so"
        )
        mock_getenv.assert_called_once_with("MINDIE_LLM_HOME_PATH")
        mock_open_fun.assert_called_once_with(self.test_pass_file)
        mock_lib.Decrypt.assert_called_once_with(
            b"fake_cipher_text",
            buffer,
            b"master_key",
            b"standby_key"
        )

    @patch('ssl.SSLContext')
    @patch('tempfile.NamedTemporaryFile')
    @patch('os.unlink')
    def test_load_ca_certificates_from_dir_success(self, mock_unlink, mock_tempfile, mock_ssl_context):
        mock_tmp_file = MagicMock()
        mock_tmp_file.name = "/tmp/test_combined.pem"
        mock_tmp_file.write = MagicMock()
        mock_tempfile.return_value = mock_tmp_file

        mock_ssl_context_instance = MagicMock()
        mock_ssl_context.return_value = mock_ssl_context_instance
        ca_dir_path = self.temp_dir
        context = ssl.SSLContext()

        CertUtil.load_ca_certificates_from_dir(ca_dir_path, context)
        mock_tempfile.assert_called_once_with(mode='w', suffix='.pem', delete=False)
        mock_ssl_context_instance.load_verify_locations.assert_called_once()
        mock_unlink.assert_called_once()


class TestTCPServer(unittest.TestCase):
    def setUp(self):
        pass
    
    def tearDown(self):
        gc.collect()

    @patch('atb_llm.utils.layerwise_disaggregated.edge_cloud_ctrl_comm.TCPServer.start_server_block', return_value=None)
    def test_init(self, mock_start):
        tcpserver = TCPServer('127.0.0.1', '1000', {})
        self.assertEqual(tcpserver.host_ip, '127.0.0.1')
        self.assertEqual(tcpserver.port, '1000')
        self.assertIsNone(tcpserver.server_socket)
        self.assertIsNone(tcpserver.clients)
        self.assertIsNone(tcpserver.clients_addr)
        self.assertFalse(tcpserver.running)
        self.assertEqual(tcpserver.recv_buf_size, LAYERWISE_DISAGGREGATED_TCP_BUFFER_SIZE)
        mock_start.assert_called_once()
    
    def test_start_server_block(self, *args):
        with patch.object(TCPServer, 'start_server_block', return_value=None) as mock_start:
            tcpserver = TCPServer('127.0.0.1', 1000, {})
            mock_start.assert_called_once()
        mock_server_socket = MagicMock()
        with patch('socket.socket', return_value=mock_server_socket):
            mock_socket = MagicMock()
            mock_address = MagicMock()
            mock_server_socket.accept.return_value = (mock_socket, mock_address)
            tcpserver.running = False
            tcpserver.start_server_block()
            self.assertTrue(tcpserver.running)
            mock_socket.setblocking.assert_called_once()

    @patch('atb_llm.utils.layerwise_disaggregated.edge_cloud_ctrl_comm.TCPServer.start_server_block', return_value=None)
    def test_send(self, *args):
        tcpserver = TCPServer('127.0.0.1', 1000, {})
        tcpserver.send('msg')
        tcpserver.clients = MagicMock()
        tcpserver.send('msg')
        tcpserver.clients.sendall.assert_called_once_with('msg'.encode('utf-8'))

    @patch('atb_llm.utils.layerwise_disaggregated.edge_cloud_ctrl_comm.TCPServer.start_server_block', return_value=None)
    def test_recv(self, *args):
        tcpserver = TCPServer('127.0.0.1', 1000, {})
        tcpserver.recv()
        tcpserver.clients = MagicMock()
        tcpserver.recv()
        tcpserver.clients.recv.assert_called_once()
    
    @patch('atb_llm.utils.layerwise_disaggregated.edge_cloud_ctrl_comm.TCPServer.start_server_block', return_value=None)
    def test_close_recv(self, *args):
        tcpserver = TCPServer('127.0.0.1', 1000, {})
        tcpserver.clients = MagicMock()
        tcpserver.server_socket = MagicMock()
        tcpserver.close_server()
        tcpserver.server_socket.close.assert_has_calls([call(), call()])


class TestTCPClient(unittest.TestCase):
    def setUp(self):
        pass
    
    def tearDown(self):
        gc.collect()

    @patch('atb_llm.utils.layerwise_disaggregated.edge_cloud_ctrl_comm.socket.socket', return_value=MagicMock())
    def test_connect_to_server_block_true(self, *args):
        tcpclient = TCPClient('127.0.0.1', '1000', {})
        res = tcpclient.connect_to_server_block()
        tcpclient.client_socket.setblocking.assert_called_once_with(False)
        self.assertIsNone(res)

    def test_send(self):
        tcpclient = TCPClient('127.0.0.1', '1000', {})
        tcpclient.send('123')
        tcpclient.client_socket = MagicMock()
        tcpclient.send('123')
        tcpclient.client_socket.sendall.assert_called_once()

    def test_recv(self):
        tcpclient = TCPClient('127.0.0.1', '1000', {})
        tcpclient.recv()
        self.assertIsNone(tcpclient.client_socket)
        tcpclient.client_socket = MagicMock()
        tcpclient.recv()
        tcpclient.client_socket.recv.assert_called_once()

    def test_is_client_connect(self):
        tcpclient = TCPClient('127.0.0.1', '1000', {})
        tcpclient.connected = True
        res = tcpclient.is_client_connected()
        self.assertTrue(res)
        tcpclient.connected = False
        res = tcpclient.is_client_connected()
        self.assertFalse(res)

    def test_disconnect(self):
        tcpclient = TCPClient('127.0.0.1', '1000', {})
        tcpclient.conneted = True
        tcpclient.client_socket = MagicMock()
        tcpclient.disconnect()
        self.assertFalse(tcpclient.connected)


class TestEdgeCloudCtrlComm(unittest.TestCase):
    def setUp(self):
        EdgeCloudCtrlComm.role = None
        EdgeCloudCtrlComm.rank = 0
        EdgeCloudCtrlComm.server_ip = ''
        EdgeCloudCtrlComm.server_port = ["2900", "9200"]

        EdgeCloudCtrlComm.prefill_server = None
        EdgeCloudCtrlComm.decode_server = None
        EdgeCloudCtrlComm.prefill_client = None
        EdgeCloudCtrlComm.decode_client = None

        EdgeCloudCtrlComm.decode_comm_finish = False
        EdgeCloudCtrlComm.prefill_comm_finish = False
        EdgeCloudCtrlComm.prefill_comm_finish_tcp_count = 0

        EdgeCloudCtrlComm.prefill_recv_msg = ''
        EdgeCloudCtrlComm.decode_recv_msg = ''
        EdgeCloudCtrlComm.prefill_send_msg = ''
        EdgeCloudCtrlComm.decode_send_msg = ''

    def tearDown(self):
        gc.collect()

    def test_init_role(self, *args):
        comm = EdgeCloudCtrlComm({})
        comm.init_role(0, '1', '1')
        self.assertEqual(comm.server_ip, '1')
        self.assertEqual(comm.role, 0)
        self.assertEqual(comm.server_port, json.loads('1'))

    def test_tcp_link(self):
        comm = EdgeCloudCtrlComm({})

        with patch.object(TCPClient, 'connect_to_server_block') as mock_con:
            comm.init_tcp_link(rank=0, role=EDGE, server_ip='', server_port='["2900", "9200"]')
            mock_con.assert_has_calls([call(), call()])

        with patch.object(TCPClient, 'connect_to_server_block') as mock_con:
            comm.init_tcp_link(rank=1, role=CLOUD, server_ip='', server_port='["2900", "9200"]')
            mock_con.assert_not_called()

        with patch.object(TCPClient, 'connect_to_server_block') as mock_con:
            comm.init_tcp_link(rank=1, role=EDGE, server_ip='', server_port='["2900", "9200"]')
            mock_con.assert_not_called()

        with patch.object(TCPClient, 'connect_to_server_block') as mock_con, \
            patch.object(TCPServer, 'start_server_block') as mock_start:
            comm.init_tcp_link(rank=0, role=CLOUD, server_ip='', server_port='["2900", "9200"]')
            mock_con.assert_not_called()
            mock_start.assert_has_calls([call(), call()])
        
        multi_nodes_infer_args = {
                'is_master': True,
                'cloud_ctrl_port': "2901",
                'cloud_ctrl_address': "2900",
                'comm_group_size': 2,
                'max_input_len': 1,
            }
        with patch.object(TCPClient, 'connect_to_server_block') as mock_con, \
            patch.object(TCPServer, 'start_server_block') as mock_start:
            comm.init_tcp_link(rank=0, role=CLOUD, server_ip='', server_port='["2900", "9200"]',
                               multi_nodes_infer_args=multi_nodes_infer_args)
            mock_con.assert_not_called()
            mock_start.assert_has_calls([call(), call()])

    def test_send_and_recv_prefill(self, *args):
        comm = EdgeCloudCtrlComm({})
        comm.server_port = ["2025", "2026"]

        with patch.object(TCPClient, 'connect_to_server_block'), \
            patch.object(TCPClient, 'send') as mock_send, \
            patch.object(TCPClient, 'recv', return_value=None) as mock_recv:
            comm.init_tcp_link(rank=0, role=EDGE, server_ip='', server_port='["2900", "9200"]')
            comm.send_prefill()
            mock_send.assert_called_once_with(comm.prefill_send_msg)
            comm.send_decode()
            mock_send.assert_has_calls([call(comm.prefill_send_msg), call(comm.decode_send_msg)])
            comm.recv_prefill()
            mock_recv.assert_called_once()
            comm.recv_decode()
            mock_recv.assert_has_calls([call(), call()])

        with patch.object(TCPClient, 'connect_to_server_block'), \
            patch.object(TCPServer, 'send') as mock_send, \
            patch.object(TCPServer, 'recv', return_value="pull1") as mock_recv, \
            patch.object(TCPServer, 'start_server_block'):
            comm.init_tcp_link(rank=0, role=CLOUD, server_ip='', server_port='["2900", "9200"]')
            comm.send_prefill()
            mock_send.assert_called_once_with(comm.prefill_send_msg)
            comm.send_decode()
            mock_send.assert_has_calls([call(comm.prefill_send_msg), call(comm.decode_send_msg)])
            comm.recv_prefill()
            mock_recv.assert_called_once()
            comm.recv_decode()
            mock_recv.assert_has_calls([call(), call()])

    def test_is_edge_cloud_ctrl_comm_success(self):
        comm = EdgeCloudCtrlComm({})
        
        comm.rank = 1
        result = comm.is_edge_cloud_ctrl_comm_success()
        self.assertTrue(result)

        comm.rank = 0
        comm.role = CLOUD
        result = comm.is_edge_cloud_ctrl_comm_success()
        self.assertTrue(result)

        comm.rank = 0
        comm.role = EDGE
        comm.prefill_client = MagicMock()
        comm.prefill_client.is_client_connected.return_value = False
        comm.decode_client = MagicMock()
        comm.decode_client.is_client_connected.return_value = False
        result = comm.is_edge_cloud_ctrl_comm_success()
        self.assertFalse(result)

    def test_shape_to_msg(self):
        comm = EdgeCloudCtrlComm({})
        self.assertIsNone(comm.shape_to_msg([]))
        self.assertIsNone(comm.shape_to_msg([1]))
        self.assertEqual(comm.shape_to_msg([4, 10]), "pull|[4, 10]|0")
        
    def test_broadcast_multi_nodes_decision(self):
        comm = EdgeCloudCtrlComm({})
        comm.rank = 0
        comm.multi_nodes_infer_enabled = True
        comm.multi_nodes_is_master = True
        comm.multi_nodes_ctrl_server = MagicMock()
        mock_send = MagicMock()
        comm.multi_nodes_ctrl_server.send = mock_send
        msg = '0'
        comm.broadcast_multi_nodes_decision(msg)
        mock_send.assert_called_once_with(msg)
        


if __name__ == '__main__':
    unittest.main()