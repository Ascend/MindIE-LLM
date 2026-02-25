#!/usr/bin/env python
# coding=utf-8
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
import subprocess
import logging
import shutil
from pathlib import Path
from setuptools.command.build_py import build_py as _build_py
from setuptools import setup, find_packages
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
import torch

logging.basicConfig(level=logging.INFO)
# SOURCE_DATE_EPOCH is used to make builds deterministic (reproducible).
os.environ["SOURCE_DATE_EPOCH"] = "315532800"  # 315532800 means 1980-01-01 00:00:00 UTC


def get_version() -> str:
    """
    Return version string.

    Priority:
    1. Environment variable MINDIE_LLM_VERSION_OVERRIDE
    2. Default version
    """
    version = os.getenv("MINDIE_LLM_VERSION_OVERRIDE", "1.0.0")
    logging.info(f"Use mindie llm version: {version}")
    return version


def use_cxx11_abi() -> str:
    """
    Return whether to use CXX11 ABI as a string ("0" or "1").
    """
    try:
        abi = torch.compiled_with_cxx11_abi()
        abi_str = str(int(bool(abi)))
        logging.info(f"Detect ABI from torch, set USE_CXX11_ABI to {abi_str}")
        return abi_str
    except Exception as e:
        logging.warning("Detect ABI from torch failed.")
        raise RuntimeError("Detect ABI from torch failed.") from e


class CustomBuildPy(_build_py):
    def run(self):
        logging.info(">>> Running build.sh to compile shared libraries...")

        project_root = Path(__file__).resolve().parent
        build_dir = project_root
        subprocess.run(
            ["/bin/bash", "build.sh", f"--use_cxx11_abi={use_cxx11_abi()}"],
            cwd=str(build_dir),
            check=True,
            shell=False
        )

        build_pkg = Path(self.build_lib) / "mindie_llm"

        shutil.copytree("output", build_pkg, dirs_exist_ok=True)
        (build_pkg / "runtime/conf").mkdir(parents=True, exist_ok=True)
        shutil.copytree("mindie_llm/runtime/conf", build_pkg / "runtime/conf", dirs_exist_ok=True)
        shutil.copytree("src/server/scripts", build_pkg / "scripts",
            dirs_exist_ok=True, ignore=shutil.ignore_patterns("set_env.sh"))
        self.copy_third_party()

        super().run()

    def copy_third_party(self):
        project_root = Path(__file__).resolve().parent
        build_pkg = Path(self.build_lib) / "mindie_llm"
        lib_dir = build_pkg / "lib"

        lib_mappings = {
            "abseil-cpp": lib_dir,
            "boost": lib_dir,
            "cares": lib_dir,
            "grpc": lib_dir / "grpc",
            "libboundscheck": lib_dir,
            "openssl": lib_dir,
            "prometheus-cpp": lib_dir,
            "protobuf": lib_dir,
            "re2": lib_dir,
            "zlib": lib_dir,
        }

        for lib_name, target_dir in lib_mappings.items():
            src_dir = project_root / "third_party" / "output" / lib_name / "lib"
            if not src_dir.exists():
                logging.warning(f"No such directory when copy it, skip: {src_dir}")
                return

            target_dir.mkdir(parents=True, exist_ok=True)
            for item in src_dir.iterdir():
                if not item.is_file():
                    continue
                dst_path = target_dir / item.name
                shutil.copy2(item, dst_path)


class BDistWheel(_bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False


setup(
    name="mindie_llm",
    version=get_version(),
    author="",
    author_email="",
    description="MindIE LLM Project",
    long_description="",
    install_requires=[],
    zip_safe=False,
    python_requires=">=3.10",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "mindie_llm_server = mindie_llm.server.main:main",
            "mindie_llm_backend = mindie_llm.connector.main:main"
        ]
    },
    cmdclass={
        "build_py": CustomBuildPy,
        "bdist_wheel": BDistWheel
    }
)
