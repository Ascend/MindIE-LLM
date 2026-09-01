# MindIE-LLM Ascend C Custom Operator (mie_ops) Adaptation Guide

<!-- md-trans-meta sourceCommit=fbb10397fd8aad33b72ee3b1e06c1b3a786f9bd6 translatedAt=2026-08-21T08:59:37.078Z pushedAt=2026-08-21T09:04:59.373Z -->

## Background

During the iterative development of MindIE versions paired with stable CANN versions, when new requirements for operators arise, you can introduce Ascend C operator source code to integrate operators. This document focuses on the integration methods and process.

## Operator Package Build

### Directory Structure

The code directory of custom operators: [src/kernels](../../../../src/kernels). The directory structure is as follows:

```text
mie_ops/                        # mie_ops operator package directory
    |_ csrc/                    # Operator source code and compilation scripts
        |_ attention/           # Attention operator source code
        |_ cmake/               # CMake scripts
        |_ common/              # Common operator utilities
        |_ mc2/                 # MC2 operator source code
        |_ scripts/             # Build scripts
        |_ utils/               # Common methods
        |_ build_aclnn.sh       # Entry script for operator compilation
        |_ build.sh             # Operator build script
        |_ CMakeLists.txt       # CMake configuration file
    |_ torch_ops_extension/     # Torch extension interfaces for operators
    |_ __init__.py               # mie_ops package initialization
build.sh                        # Entry script for building the mie_ops package
setup.py                        # Setup script for the mie_ops wheel package
```

### Build and Installation

#### Separate Build and Installation of the Operator Package

```shell
# Step 1: Operator compilation depends on the third-party components uniformly managed by MindIE. Compile the third-party components first for the initial compilation
bash build.sh 3rd

# Step 2: Compile the operator package
cd src/kernels
bash build.sh
# Path of the operator package generated after compilation: src/kernels/dist. The package name distinguishes the chip
# A2：mie_ops_ascend910b-1.0-cp311-cp311-linux_aarch64.whl
# A3：mie_ops_ascend910_93-1.0-cp311-cp311-linux_aarch64.whl

# Step 3: Install the operator package
pip install dist/mie_ops*.whl --force-reinstall
# Directory after installation, using the default installation path as an example.
# A2：/usr/local/lib/python3.11/site-packages/mie_ops_ascend910b
# A3：/usr/local/lib/python3.11/site-packages/mie_ops_ascend910_93
```

#### MindIE-LLM Full Build and Installation

To build the complete MindIE-LLM package, refer to the [MindIE-LLM Full Package Build Guide](../build_guide_llm.md). This document provides supplementary details specifically for the mie_ops operator package included within the full package.

**Operator package directory after installation, using the default installation path as an example**

- **.whl package**

1. A2: `/usr/local/lib/python3.11/site-packages/mindie_llm/lib/mie_ops_ascend910b`

2. A3: `/usr/local/lib/python3.11/site-packages/mindie_llm/lib/mie_ops_ascend910_93`

- **.run package**

1. A2: `/usr/local/Ascend/mindie/latest/mindie-llm/lib/mie_ops_ascend910b`

2. A3: `/usr/local/Ascend/mindie/latest/mindie-llm/lib/mie_ops_ascend910_93`

**NOTE**

1. When developers build the MindIE-LLM full package on their own environment, the environment type is automatically identified and the corresponding operator package is compiled (depending on the corresponding HDK package and CANN package installed in the environment). Therefore, only the operator package for the current environment type is included.

2. The MindIE-LLM full package built by CI contains operator packages for all chip types currently supported by MindIE-LLM, and the appropriate one is dynamically selected and imported at runtime based on the environment type.

3. Because the mie_ops operator package needs to be recompiled and rebuilt only in a few scenarios, such as integrating a new custom operator or updating operators, it is treated as a second-party component of MindIE-LLM. The mie_ops operator package is compiled only when the MindIE-LLM full package is built for the first time. If the code under the `src/kernels` directory is updated later and the operator package needs to be recompiled, delete the existing operator package under the `src/kernels/dist` directory first so that the mie_ops operator package is recompiled when the MindIE-LLM full package is built.

## New Operator Integration

Using the MOE large fusion operator dispatch_ffn_combine as an example, this section describes the new operator integration process.

### New Operator Integration Build Project

#### Preparing Operator Source Code

Place the operator source code and the corresponding `CMakeLists.txt` in the corresponding directory, as shown in the following example:

```text
mie_ops/                                 # mie_ops operator package root
    |_ csrc/                             # operator source and build scripts
        |_ mc2/                           # mc2 operator sources
            |_ dispatch_ffn_combine/      # dispatch_ffn_combine operator sources
```

#### Adding the Operator to the Compilation List

In [build_aclnn.sh](../../../../src/kernels/mie_ops/csrc/build_aclnn.sh), add the operator to the compilation list of the required target chip. For example, the dispatch_ffn_combine operator is supported only on A3, so add it to the A3 compilation list.

```shell
if [[ "$SOC_VERSION" =~ ^ascend910b ]]; then
    # ASCEND910B (A2) series
    ...
elif [[ "$SOC_VERSION" =~ ^ascend910_93 ]]; then
    # ASCEND910C (A3) series
    ...
    CUSTOM_OPS_ARRAY=(
        "dispatch_ffn_combine" # Add the operator to this compilation list
    )
    ...
else
    # others
    # currently, no custom aclnn ops for other series
    exit 0
fi
```

#### Registering the torch Extension Interface

Directory structure reference:

```text
mie_ops/                                 # mie_ops operator package root
    |_ torch_ops_extension/              # PyTorch operator extension interface
        |_ npu_dispatch_ffn_combine.cpp  # forward interface registration for dispatch_ffn_combine
        |_ ops_def_registration.cpp      # PyTorch operator interface registration
```

- Step 1: Add torch interface definition registration for the new custom operator in [ops_def_registration.cpp](../../../../src/kernels/mie_ops/torch_ops_extension/ops_def_registration.cpp).

```cpp
// step1, add the definition for the new custom operator
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

- Step 2-5: Add a .cpp file named after the operator, such as [npu_dispatch_ffn_combine.cpp](../../../../src/kernels/mie_ops/torch_ops_extension/npu_dispatch_ffn_combine.cpp), to register the forward implementation interface for the npu and meta devices respectively.

```cpp
#include <iostream>
#include <torch/library.h>
#include "ops_common.h"

namespace mie_ops {
using namespace at_npu::native;

// Step 2: Implement the forward interface for the NPU device
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

// Step 3: Implement the forward interface for the META device
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

// Step 4: Register the forward implementation for the NPU device
TORCH_LIBRARY_IMPL(mie_ops, PrivateUse1, m) {
    m.impl("npu_dispatch_ffn_combine", &mie_ops::npu_dispatch_ffn_combine_npu);
}

// Step 5: Register the forward implementation for the META device
TORCH_LIBRARY_IMPL(mie_ops, Meta, m) {
    m.impl("npu_dispatch_ffn_combine", &mie_ops::npu_dispatch_ffn_combine_meta);
}
```

#### Single-Operator UT Verification

After the operator package is successfully compiled and installed, you can refer to the test cases in the [tests/pythontest/test_kernels](../../../../tests/pythontest/test_kernels) directory of the code repository and write a single-operator UT to preliminarily verify whether the new operator is correctly packaged and whether its basic functions are correct.

### New Operator Integration into a Model

#### Importing the Operator Module

The `mindie-llm` package built by the current CI installs operator packages for different chip types (A2/A3) at the same time. Import them in the following way to support adaptively selecting the operator package that matches the actual chip type of the runtime environment.

```python
from mindie_llm.runtime.ops import mie_ops
```

#### Operator Invocation

Invoke the operator through `torch.ops.mie_ops.{operator interface name}`, where `{operator interface name}` is the interface name defined in `ops_def_registration.cpp`. For the meaning and constraints of each operator parameter, refer to the operator interface documentation. Taking the dispatch_ffn_combine operator as an example, the invocation method is as follows:

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

#### Full-network Verification

Verify the functionality, precision, and performance of the model after the new operator is integrated.

## FAQ

- **Q: What Should I Do When the Custom Operator Name Conflicts with an Existing Operator Name in the Cann Package, Causing Incorrect Operator Invocation?**

  A: You can avoid this by modifying the custom operator name. For example, add a `_mie` or `Mie` suffix according to the existing naming style to indicate that it is a mie_ops custom operator and avoid conflicts; note that you need to modify the operator directory name, file name, OP_NAME and OPTYPE in `CMakeLists.txt`, and the operator name in the operator source code at the same time.
