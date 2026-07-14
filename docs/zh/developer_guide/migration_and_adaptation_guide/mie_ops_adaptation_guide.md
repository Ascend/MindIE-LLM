# MindIE-LLM Ascend C 自定义算子 (mie_ops) 适配指南

## 背景

MindIE 版本配套稳定 CANN 版本迭代开发过程中，涉及对算子的新诉求，可以自行引入 Ascend C 算子源码来进行算子接入，本文档重点介绍接入方法和流程。

## 算子包构建

### 目录结构

自定义算子所在代码目录：[src/kernels](../../../../src/kernels)，目录结构如下：

```text
mie_ops/                        # mie_ops 算子包目录
    |_ csrc/                    # 算子源码和编译脚本目录
        |_ attention/           # attention 算子源码目录
        |_ cmake/               # cmake 脚本目录
        |_ common/              # 算子公共功能目录
        |_ mc2/                 # mc2 算子源码目录
        |_ scripts/             # 构建脚本所在目录
        |_ utils/               # 公共方法目录
        |_ build_aclnn.sh       # 算子编译入口脚本
        |_ build.sh             # 算子编译脚本
        |_ CMakeLists.txt       # CMake 配置文件
    |_ torch_ops_extension/     # 算子 torch 扩展接口目录
    |_ __init__.py              # mie_ops 算子包初始化代码
build.sh                        # mie_ops 算子包构建入口脚本
setup.py                        # mie_ops 算子 whl 包 setup 脚本
```

### 编译安装

#### 算子包单独编译安装

```shell
# Step 1：算子编译依赖 MindIE 统一管理的三方件，首次编译需要先行编译三方件
bash build.sh 3rd

# Step 2：编译算子包
cd src/kernels
bash build.sh
# 编译完成后生成的算子包路径：src/kernels/dist，包名区分芯片
# A2：mie_ops_ascend910b-1.0-cp311-cp311-linux_aarch64.whl
# A3：mie_ops_ascend910_93-1.0-cp311-cp311-linux_aarch64.whl

# Step 3：安装算子包
pip install dist/mie_ops*.whl --force-reinstall
# 安装后目录，以默认安装路径为例
# A2：/usr/local/lib/python3.11/site-packages/mie_ops_ascend910b
# A3：/usr/local/lib/python3.11/site-packages/mie_ops_ascend910_93
```

#### MindIE-LLM 整包编译安装

可参考 [MindIE-LLM 整包编译安装参考文档](../build_guide_llm.md)编译 MindIE-LLM 整包，下述对整包内的 mie_ops 算子包进行补充说明。

**安装后算子包目录，以默认安装路径为例**

- **whl 包**

1. A2: /usr/local/lib/python3.11/site-packages/mindie_llm/lib/mie_ops_ascend910b

2. A3: /usr/local/lib/python3.11/site-packages/mindie_llm/lib/mie_ops_ascend910_93

- **run 包**

1. A2: /usr/local/Ascend/mindie/latest/mindie-llm/lib/mie_ops_ascend910b

2. A3: /usr/local/Ascend/mindie/latest/mindie-llm/lib/mie_ops_ascend910_93

**注意点**

1. 开发人员在环境上自行编译 MindIE-LLM 整包时，会自动识别环境类型并编译对应的算子包（依赖环境上装有对应类型的 HDK 包和 CANN 包），因此仅包含当前环境类型的算子包。
2. CI 构建的 MindIE-LLM 整包，包含当前 MindIE-LLM 已支持的所有芯片类型的算子包，运行时根据环境类型动态选择导入。
3. 由于 mie_ops 算子包仅在接入新的自定义算子和更新算子等少量场景需要重新进行编译构建，因此作为 MindIE-LLM 的二方件看待，仅在第一次编译 MindIE-LLM 整包时会对 mie_ops 算子包进行编译；后续如果更新了 src/kernels 目录下的代码，需要重新编译算子包时，请先删除 src/kernels/dist 目录下已存在的算子包，使得 MindIE-LLM 整包编译时触发 mie_ops 算子包重新编译。

## 新算子接入

以 MOE 大融合算子 dispatch_ffn_combine 为例，介绍新算子接入流程

### 新算子接入构建工程

#### 准备算子源码

将算子源码和对应的 CMakeLists.txt 放置到对应目录，参考如下所示：

```text
mie_ops/                                # mie_ops 算子包目录
    |_ csrc/                            # 算子源码和编译脚本目录
        |_ mc2/                         # mc2 算子源码目录
            |_ dispatch_ffn_combine/    # dispatch_ffn_combine 算子源码目录
```

#### 将算子加入编译列表

在 [build_aclnn.sh](../../../../src/kernels/mie_ops/csrc/build_aclnn.sh) 中，将算子加入所需编译芯片的编译列表，例如 dispatch_ffn_combine 算子仅在 A3 上支持，加入 A3 的编译列表中

```shell
if [[ "$SOC_VERSION" =~ ^ascend910b ]]; then
    # ASCEND910B (A2) series
    ...
elif [[ "$SOC_VERSION" =~ ^ascend910_93 ]]; then
    # ASCEND910C (A3) series
    ...
    CUSTOM_OPS_ARRAY=(
        "dispatch_ffn_combine" # 将算子加入该编译列表
    )
    ...
else
    # others
    # currently, no custom aclnn ops for other series
    exit 0
fi
```

#### 注册 torch 扩展接口

目录结构参考：

```text
mie_ops/                                   # mie_ops 算子包目录
    |_ torch_ops_extension/                # 算子 torch 扩展接口目录
        |_ npu_dispatch_ffn_combine.cpp    # dispatch_ffn_combine 算子前向接口注册代码
        |_ ops_def_registration.cpp        # 算子 torch 接口注册文件
```

- Step 1：在 [ops_def_registration.cpp](../../../../src/kernels/mie_ops/torch_ops_extension/ops_def_registration.cpp) 中为新增自定义算子添加 torch 接口定义注册。

```cpp
// step1, 为新增自定义算子添加定义
TORCH_LIBRARY(mie_ops, m) {
    m.def(
        "npu_dispatch_ffn_combine("
        "Tensor x, "
        "Tensor[] weight1, "
        "Tensor[] weight2, "
        "Tensor expert_idx, "
        "Tensor[] scale1, "
        "Tensor[] scale2, "
        "Tensor probs, "
        "str group, "
        "int max_output_size, "
        "Tensor! out"
        ") -> Tensor");
}
```

- Step 2-5：新增以算子命名的的 cpp 文件，如 [npu_dispatch_ffn_combine.cpp](../../../../src/kernels/mie_ops/torch_ops_extension/npu_dispatch_ffn_combine.cpp) ，为 npu 和 meta 设备分别注册前向实现接口。

```cpp
#include <iostream>
#include <torch/library.h>
#include "ops_common.h"

namespace mie_ops {
using namespace at_npu::native;

// step2, 为NPU设备实现前向接口
at::Tensor& npu_dispatch_ffn_combine_npu(
    const at::Tensor& x,
    const at::TensorList& weight1,
    const at::TensorList& weight2,
    const at::Tensor& expert_idx,
    const at::TensorList& scale1,
    const at::TensorList& scale2,
    const at::Tensor& probs,
    c10::string_view group,
    int64_t max_output_size,
    at::Tensor& out
) {
    char *group_ep_ptr = const_cast<char *>(group.data());
    EXEC_NPU_CMD_V1(aclnnDispatchFFNCombine,
                    x,
                    weight1,
                    weight2,
                    expert_idx,
                    scale1,
                    scale2,
                    probs,
                    group_ep_ptr,
                    max_output_size,
                    out);
    return out;
}

// step3, 为META设备实现前向接口
at::Tensor& npu_dispatch_ffn_combine_meta(
    const at::Tensor& x,
    const at::TensorList& weight1,
    const at::TensorList& weight2,
    const at::Tensor& expert_idx,
    const at::TensorList& scale1,
    const at::TensorList& scale2,
    const at::Tensor& probs,
    c10::string_view group,
    int64_t max_output_size,
    at::Tensor& out
) {
    return out;
}

}  // namespace mie_ops

// step4, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(mie_ops, PrivateUse1, m) {
    m.impl("npu_dispatch_ffn_combine", &mie_ops::npu_dispatch_ffn_combine_npu);
}

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(mie_ops, Meta, m) {
    m.impl("npu_dispatch_ffn_combine", &mie_ops::npu_dispatch_ffn_combine_meta);
}
```

#### 单算子 UT 验证

成功编译并安装算子包后，可以参考代码仓 [tests/pythontest/test_kernels](../../../../tests/pythontest/test_kernels) 目录下的测试用例，写一个单算子 UT 初步验证新算子是否正确打包，以及基本功能是否正确

### 新算子接入模型

#### 算子模块导入

当前 CI 构建的 mindie-llm 包同时安装了不同芯片类型的算子包 (A2/A3)，通过下述方式导入支持自适应选择运行环境实际芯片类型的算子包。

```python
from mindie_llm.runtime.ops import mie_ops
```

#### 算子调用

通过 torch.ops.mie_ops.{算子接口名} 的方式调用算子，其中 {算子接口名} 即为 ops_def_registration.cpp 中定义的接口名称，算子各参数的含义和约束，请参考算子接口文档，以 dispatch_ffn_combine 算子为例，调用方式如下：

```python
torch.ops.mie_ops.npu_dispatch_ffn_combine(
    x=...,
    weight1=...,
    weight2=...,
    expert_idx=...,
    scale1=...,
    scale2=...,
    probs=...,
    group=...,
    max_output_size=...,
    out=...,
)
```

#### 整网验证

整网验证新算子接入后的功能、精度和性能。

## FAQ

- **Q: 当自定义算子名称和 CANN 包里面已有算子名称冲突导致算子调用错误时如何处理？**

  A: 可以通过修改自定义算子名称来进行规避（例如，根据已有命名风格加 _mie 或者 Mie 后缀来说明是 mie_ops 自定义算子，避免冲突；注意需要同时修改算子的目录名、文件名、CMakeLists.txt 中的 OP_NAME 和 OPTYPE、以及算子源码中的算子名。
