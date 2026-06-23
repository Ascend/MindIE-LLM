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

import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py


os.environ["SOURCE_DATE_EPOCH"] = "0"

parser = argparse.ArgumentParser(description="MindIE LLM Setup Parameters")
parser.add_argument("--setup_cmd", type=str, default="bdist_wheel")
parser.add_argument("--version", type=str, default="1.0.0")

args = parser.parse_args()
sys.argv = [sys.argv[0], args.setup_cmd]
mindie_llm_version = args.version


# 定义.proto文件路径（假设在项目根目录的protos/文件夹下）
class BuildPyCommand(build_py):
    def run(self):
        # 获取当前环境变量
        env = os.environ.copy()
        # 添加protoc所需的动态库路径到LD_LIBRARY_PATH
        libdir = "lib64" if os.path.exists("/etc/openEuler-release") else "lib"
        protobuf_lib_path = os.path.abspath(f"./third_party/output/protobuf/{libdir}")
        absl_lib_path = os.path.abspath(f"./third_party/output/abseil-cpp/{libdir}")
        if "LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = f"{protobuf_lib_path}:{absl_lib_path}:{env['LD_LIBRARY_PATH']}"
        else:
            env["LD_LIBRARY_PATH"] = f"{protobuf_lib_path}:{absl_lib_path}"
        subprocess.run(
            [
                "./third_party/output/protobuf/bin/protoc",  # 使用相对路径调用protoc
                "--experimental_allow_proto3_optional",
                "--python_out=./mindie_llm/connector/common/",
                "--proto_path=./proto/",
                "model_execute_data.proto",
            ],
            env=env,
            check=True,
        )

        build_pkg = Path(self.build_lib) / "mindie_llm"
        (build_pkg / "runtime/conf").mkdir(parents=True, exist_ok=True)
        shutil.copytree("mindie_llm/runtime/conf", build_pkg / "runtime/conf", dirs_exist_ok=True)

        # 继续执行默认的build_py逻辑
        super().run()


setup(
    name="mindie_llm",
    version=mindie_llm_version,
    author="",
    author_email="",
    description="MindIE LLM Project",
    long_description="",
    package_dir={"mindie_llm": "mindie_llm"},
    install_requires=[],
    package_data={"": ["*.xlsx", "*.h5", "*.csv", "*.so", "*.avsc", "*.xml", "*.pkl", "*.sql", "*.ini"]},
    zip_safe=False,
    python_requires=">=3.10",
    cmdclass={
        "build_py": BuildPyCommand,
    },
    include_package_data=True,
    entry_points={"console_scripts": ["mindie_llm_backend = mindie_llm.connector.main:main"]},
)
