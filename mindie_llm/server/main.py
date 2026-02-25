#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import sys
import sysconfig
import importlib.util
from pathlib import Path
from mindie_llm.utils.log.logging import logger


def _get_pkg_dir(pkg_name: str) -> Path:
    spec = importlib.util.find_spec(pkg_name)
    if spec is None or spec.origin is None:
        raise ImportError(f"{pkg_name} is not installed")
    return Path(spec.origin).resolve().parent


def _prepend_ld_library_path(env: dict, paths: list[Path]) -> None:
    """Prepend existing directories to LD_LIBRARY_PATH, warn if missing."""
    valid_paths = []

    for p in paths:
        if p.is_dir():
            valid_paths.append(str(p))
        else:
            logger.warning(f"LD_LIBRARY_PATH entry does not exist, skipped: {p}")

    if not valid_paths:
        return

    old = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(valid_paths + ([old] if old else []))


def main():
    pkg_root = Path(__file__).resolve().parents[1]

    daemon_path = pkg_root / "bin" / "mindieservice_daemon"
    if not daemon_path.is_file():
        raise RuntimeError(f"mindieservice_daemon not found at {daemon_path}")

    # Set configure file permission to meet safety requirement
    config_path = pkg_root / "conf" / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"config.json not found at {config_path}")

    # Set environment variables
    env = os.environ.copy()
    env["MINDIE_LLM_HOME_PATH"] = str(pkg_root)

    lib_str = "lib"
    lib_dir = pkg_root / lib_str
    if not lib_dir.is_dir():
        raise RuntimeError(f"Lib directory not found at {lib_dir}")

    # torch
    torch_dir = _get_pkg_dir("torch")
    torch_lib = torch_dir / lib_str
    torch_libs = torch_dir.parent / "torch.libs"  # torch.libs is a sibling of torch/

    # atb_llm
    atb_llm_dir = _get_pkg_dir("atb_llm")
    atb_llm_lib = atb_llm_dir / lib_str

    _prepend_ld_library_path(
        env,
        [
            torch_lib,
            torch_libs,
            atb_llm_lib,
            lib_dir,
            lib_dir / "grpc",
        ],
    )

    env["PYTHONPATH"] = f"{lib_dir}:{env.get('PYTHONPATH', '')}"

    os.execve(
        str(daemon_path),
        [str(daemon_path)] + sys.argv[1:],
        env,
    )


if __name__ == "__main__":
    main()
