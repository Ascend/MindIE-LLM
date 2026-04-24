set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_SKIP_RPATH TRUE)

get_ABI_option_value()

# 通用编译选项
set(_COMMON_FLAGS_LIST
    -ggdb                                      # 生成 GDB 调试信息
    -pipe                                      # 加快编译，使用管道而非临时文件
    -fstack-protector-strong                   # 强化栈保护，检测栈溢出
    -fPIE                                      # 生成位置无关代码（可执行文件 ASLR 必需）
    -fno-common                                # 禁止多重定义的全局变量被合并，避免弱符号问题
    -D_FORTIFY_SOURCE=2                        # glibc 编译期/运行时安全检查（格式化、缓冲区）
    -Wall                                      # 常规警告
    -Wextra                                    # 更多额外警告
    -Wfloat-equal                              # 浮点比较警告

    -Wformat                                   # 检查 printf/scanf 格式化问题
    -Wformat-security                          # 禁止不安全的格式化字符串
    -Wstack-protector                          # 确保栈保护启用
    -Wl,-z,relro                               # 链接器：启用 RELRO，部分 GOT 只读
    -Wl,-z,now                                 # 链接器：程序启动即绑定符号（Full RELRO）
    -Wl,-z,noexecstack                         # 链接器：禁止可执行栈
    -Wl,--as-needed                            # 链接器：仅加载真正需要的库，减少攻击面
)

# C 语言特有选项
set(_C_FLAGS_LIST
    -Wimplicit-function-declaration            # 禁止使用未声明的函数
    -Wmissing-prototypes                       # 要求函数在头文件声明
    -Wstrict-prototypes                        # C 函数参数必须显式类型（不允许隐式 int）
    -Wold-style-definition                     # 禁止 K&R 风格函数定义
    -Wshadow                                   # 警告变量名遮蔽（容易出现逻辑错误）
    -Wpointer-arith                            # 不安全的指针算术警告
)

# C++ 特有选项
set(_CXX_FLAGS_LIST
    -fexceptions                               # 启用 C++ 异常（大多数项目都需要）
    -Wnon-virtual-dtor                         # 基类没有虚析构时警告（防止内存泄露）
    -Wdelete-non-virtual-dtor                  # 删除非虚析构基类指针导致 UB 时警告
    -Woverloaded-virtual                       # 检测覆盖隐藏基类虚函数
    -Wzero-as-null-pointer-constant            # 禁止用 0/NULL 代替 nullptr
    -Wold-style-cast                           # 禁止 C 风格强制类型转换
    -Wduplicated-cond                          # 重复条件检测（逻辑 bug）
    -Wduplicated-branches                      # if/else 分支冗余检测
)

list(JOIN _COMMON_FLAGS_LIST " " _COMMON_FLAGS)
list(JOIN _C_FLAGS_LIST " " _C_FLAGS)
list(JOIN _CXX_FLAGS_LIST " " _CXX_FLAGS)
# 应用全局 C / C++ 编译选项
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${_COMMON_FLAGS} ${_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_COMMON_FLAGS} ${_CXX_FLAGS}")
if(USE_CXX11_ABI)
    add_definitions(-U_GLIBCXX_USE_CXX11_ABI -D_GLIBCXX_USE_CXX11_ABI=1)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

# 链接器选项
set(_LD_FLAGS_GLOBAL_LIST
    -rdynamic                                  # 保留符号（调试/插件系统需要）
    -ldl                                       # 显式链接 libdl（dlopen/dlsym）
    -Wl,-z,relro                               # RELRO（只读 GOT）
    -Wl,-z,now                                 # Full RELRO（立即绑定）
    -Wl,-z,noexecstack                         # 栈不可执行
    -Wl,--as-needed                            # 只加载需要的库
)
list(JOIN _LD_FLAGS_GLOBAL_LIST " " LD_FLAGS_GLOBAL)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${LD_FLAGS_GLOBAL}")
set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS} ${LD_FLAGS_GLOBAL} -pie")  # 可执行文件启用 PIE

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR "$ENV{MINDIE_ENABLE_PROF}" MATCHES "^1$")
    add_definitions(-DENABLE_PROF)             # 向代码注入宏 ENABLE_PROF
endif()

if(USE_FUZZ_TEST OR DOMAIN_LAYERED_TEST)
    enable_testing()
    add_compile_definitions(UT_ENABLED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
endif()

if(USE_FUZZ_TEST)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize-coverage=trace-pc")   # fuzz 用代码覆盖
    add_definitions(-DFUZZ_TEST)
endif()

# 第三方编译选项
set(THIRD_PARTY_C_FLAGS
    -fstack-protector-strong
    -D_FORTIFY_SOURCE=2
    -O2
    -ftrapv
    -Wl,-z,relro
    -Wl,-z,now
    -s
)
get_architecture(ARCH)
if(${ARCH} STREQUAL "aarch64")
    list(APPEND THIRD_PARTY_C_FLAGS "-march=armv8-a+crc")
endif()
set(THIRD_PARTY_CXX_FLAGS ${THIRD_PARTY_C_FLAGS})
list(APPEND THIRD_PARTY_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI}")
