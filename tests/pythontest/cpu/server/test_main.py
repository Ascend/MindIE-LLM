import os
import sys
import unittest
from unittest import mock
from pathlib import Path

from mindie_llm.server.main import main


class TestServerMain(unittest.TestCase):

    @mock.patch("mindie_llm.server.main._get_pkg_dir")
    @mock.patch("mindie_llm.server.main.os.execve")
    @mock.patch("mindie_llm.server.main.Path.is_dir")
    @mock.patch("mindie_llm.server.main.Path.is_file")
    @mock.patch("mindie_llm.server.main.os.environ", new_callable=dict)
    def test_main_success_no_args(
        self,
        mock_environ,
        mock_is_file,
        mock_is_dir,
        mock_execve,
        mock_get_pkg_dir,
    ):
        # daemon + config.json exist
        mock_is_file.side_effect = [True, True]
        mock_is_dir.return_value = True

        # Fake package locations
        fake_site = Path("/fake/site-packages")
        mock_get_pkg_dir.side_effect = [
            fake_site / "torch",     # torch
            fake_site / "atb_llm",   # atb_llm
        ]

        mock_environ.update({
            "LD_LIBRARY_PATH": "/old/ld",
            "PYTHONPATH": "/old/python",
        })

        with mock.patch.object(sys, "argv", ["mindie_llm_server"]):
            main()

        mock_execve.assert_called_once()
        exec_path, exec_argv, exec_env = mock_execve.call_args[0]

        pkg_root = Path(__file__).resolve().parents[4] / "mindie_llm"
        daemon_path = pkg_root / "bin" / "mindieservice_daemon"
        lib_dir = pkg_root / "lib"

        self.assertEqual(exec_path, str(daemon_path))
        self.assertEqual(exec_argv, [str(daemon_path)])

        # env checks
        self.assertEqual(exec_env["MINDIE_LLM_HOME_PATH"], str(pkg_root))
        self.assertIn(str(lib_dir), exec_env["LD_LIBRARY_PATH"])
        self.assertIn("grpc", exec_env["LD_LIBRARY_PATH"])
        self.assertIn(str(lib_dir), exec_env["PYTHONPATH"])

    @mock.patch("mindie_llm.server.main._get_pkg_dir")
    @mock.patch("mindie_llm.server.main.Path.is_file")
    def test_daemon_missing_raises(self, mock_is_file, mock_get_pkg_dir):
        mock_is_file.return_value = False

        with self.assertRaises(RuntimeError) as ctx:
            main()

        self.assertIn("mindieservice_daemon not found", str(ctx.exception))

    @mock.patch("mindie_llm.server.main._get_pkg_dir")
    @mock.patch("mindie_llm.server.main.Path.is_file")
    @mock.patch("mindie_llm.server.main.Path.is_dir")
    def test_lib_dir_missing_raises(
        self,
        mock_is_dir,
        mock_is_file,
        mock_get_pkg_dir,
    ):
        mock_is_file.side_effect = [True, True]
        mock_is_dir.return_value = False

        with self.assertRaises(RuntimeError) as ctx:
            main()

        self.assertIn("Lib directory not found", str(ctx.exception))
