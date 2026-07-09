# ============================================================
# Dockerfile — MindIE multi-stage image
# ============================================================
# Build args drive OS, chip, arch, python, CANN, PTA selection.
# A5 is not supported.
# Usage:
#   docker build --build-arg OS_TYPE=ubuntu --build-arg CHIP=910 \
#     --build-arg ARCH=x86_64 --build-arg PYTHON_VERSION=3.11.10 \
#     ... -f Dockerfile -t mindie:latest downloads/
# ============================================================

ARG OS_TYPE=ubuntu

# ================================================================
# Stage 1a: Ubuntu base (compiled python)
# ================================================================
FROM ubuntu:24.04 AS base-ubuntu

ARG PYTHON_VERSION
ARG PYTHON_MAJOR_VERSION
ARG PYTHON_SHORT_VERSION

# Copy Python source tarball (only dependency needed by base-ubuntu)
COPY downloads/Python-${PYTHON_VERSION}.tar.xz /opt/package/

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common
ENV LANG=C.UTF-8
ENV PATH=/usr/local/bin:$PATH

RUN sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.huaweicloud.com/ubuntu/|g' \
        -e 's|http://security.ubuntu.com/ubuntu/|http://mirrors.huaweicloud.com/ubuntu/|g' \
        -e 's|http://ports.ubuntu.com/ubuntu-ports/|http://mirrors.huaweicloud.com/ubuntu-ports/|g' \
        /etc/apt/sources.list.d/ubuntu.sources && \
    apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    sed -i 's|http://mirrors.huaweicloud.com/|https://mirrors.huaweicloud.com/|g' /etc/apt/sources.list.d/ubuntu.sources && \
    apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates netbase tzdata wget vim libjemalloc2 && \
    rm -rf /var/lib/apt/lists/*

# compile python from source
RUN set -eux; \
    savedAptMark="$(apt-mark showmanual)"; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        dpkg-dev gcc gnupg libbluetooth-dev libbz2-dev libc6-dev libdb-dev \
        libffi-dev libgdbm-dev liblzma-dev libncursesw5-dev libreadline-dev \
        libsqlite3-dev libssl-dev make tk-dev uuid-dev xz-utils zlib1g-dev; \
    mkdir -p /opt/package /usr/src/python; \
    cd /opt/package; \
    ls -l; \
    tar --extract --directory /usr/src/python --strip-components=1 --file Python-${PYTHON_VERSION}.tar.xz; \
    cd /usr/src/python; \
    gnuArch="$(dpkg-architecture --query DEB_BUILD_GNU_TYPE)"; \
    ./configure \
        --build="$gnuArch" --enable-loadable-sqlite-extensions \
        --enable-optimizations --enable-option-checking=fatal \
        --enable-shared --with-lto --with-ensurepip; \
    EXTRA_CFLAGS="$(dpkg-buildflags --get CFLAGS)"; \
    LDFLAGS="$(dpkg-buildflags --get LDFLAGS)"; \
    LDFLAGS="${LDFLAGS:--Wl},--strip-all"; \
    make -j "$(nproc)" "EXTRA_CFLAGS=${EXTRA_CFLAGS:-}" "LDFLAGS=${LDFLAGS:-}"; \
    rm python; \
    make -j "$(nproc)" "EXTRA_CFLAGS=${EXTRA_CFLAGS:-}" "LDFLAGS=${LDFLAGS:--Wl},-rpath='\$\$ORIGIN/../lib'" python; \
    make install; \
    cd /; rm -rf /usr/src/python; \
    find /usr/local -depth \
        \( \( -type d -a \( -name test -o -name tests -o -name idle_test \) \) \
           -o \( -type f -a \( -name '*.pyc' -o -name '*.pyo' -o -name 'libpython*.a' \) \) \
        \) -exec rm -rf '{}' +; \
    ldconfig; \
    apt-mark auto '.*' > /dev/null; \
    apt-mark manual $savedAptMark; \
    find /usr/local -type f -executable -not \( -name '*tkinter*' \) -exec ldd '{}' ';' \
        | awk '/=>/ { so = $(NF-1); if (index(so, "/usr/local/") == 1) { next }; gsub("^/(usr/)?", "", so); printf "*%s\n", so }' \
        | sort -u | xargs -r dpkg-query --search 2>/dev/null \
        | awk -F: '/^[^:]*:/ && !/^diversion/ {print $1 }' | sort -u | xargs -r apt-mark manual; \
    apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false; \
    rm -rf /var/lib/apt/lists/*; \
    mkdir -p /root/.pip && \
    echo "[global]" > /root/.pip/pip.conf && \
    echo "index-url=https://repo.huaweicloud.com/repository/pypi/simple" >> /root/.pip/pip.conf && \
    echo "trusted-host=repo.huaweicloud.com" >> /root/.pip/pip.conf && \
    echo "timeout=120" >> /root/.pip/pip.conf && \
    pip3 install --disable-pip-version-check --no-cache-dir --no-compile 'setuptools==65.5.1' wheel; \
    pip3 --version

RUN set -eux; \
    for src in idle3 pip3 pydoc3 python3 python3-config; do \
        dst="$(echo "$src" | tr -d 3)"; \
        [ -s "/usr/local/bin/$src" ]; \
        [ ! -e "/usr/local/bin/$dst" ]; \
        ln -svT "$src" "/usr/local/bin/$dst"; \
    done

# ================================================================
# Stage 1b: OpenEuler base (pre-installed python)
# ================================================================
FROM hub.oepkgs.net/openeuler/openeuler:24.03-lts-sp3 AS base-openeuler

ENV LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/lib64
ENV TZ=Asia/Shanghai

RUN dnf update -y && \
    dnf --setopt=sslverify=false install -y python3-pip shadow-utils wget vim util-linux \
        findutils python3-devel pciutils jemalloc diffutils kmod grubby lsscsi && \
    ln -sf /usr/bin/pip3.11 /usr/bin/pip && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    mkdir -p /root/.pip && \
    echo "[global]" > /root/.pip/pip.conf && \
    echo "index-url=https://repo.huaweicloud.com/repository/pypi/simple" >> /root/.pip/pip.conf && \
    echo "trusted-host=repo.huaweicloud.com" >> /root/.pip/pip.conf && \
    echo "timeout=120" >> /root/.pip/pip.conf && \
    pip install --no-cache-dir wheel

# ================================================================
# Stage 2: Base (dynamic OS selection)
# ================================================================
FROM base-${OS_TYPE} AS base

ARG OS_TYPE
ARG CHIP
ARG ARCH

# Copy all downloaded packages into the container
COPY downloads/*.run downloads/*.whl downloads/*.tar.xz /opt/package/

# ================================================================
# Stage 3: CANN
# ================================================================
FROM base AS cann

ARG CANN_VERSION
ARG CANN_DEVICE
ARG ARCH

ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64/plugin/nnengine:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/${ARCH}:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/tools/aml/lib64:${ASCEND_TOOLKIT_HOME}/tools/aml/lib64/plugin:$LD_LIBRARY_PATH
ENV PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe
ENV PATH=${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:${ASCEND_TOOLKIT_HOME}/tools/ccec_compiler/bin:$PATH
ENV ASCEND_AICPU_PATH=${ASCEND_TOOLKIT_HOME}
ENV ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp
ENV TOOLCHAIN_HOME=${ASCEND_TOOLKIT_HOME}/toolkit
ENV ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}

# NNAL
ENV ATB_HOME_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1
ENV LD_LIBRARY_PATH=${ATB_HOME_PATH}/lib:${ATB_HOME_PATH}/examples:${ATB_HOME_PATH}/tests/atbopstest:$LD_LIBRARY_PATH
ENV PATH=${ATB_HOME_PATH}/bin:$PATH
ENV ASDOPS_HOME_PATH=${ATB_HOME_PATH}

ENV ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=0
ENV ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE=0
ENV ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE=0
ENV ATB_OPSRUNNER_KERNEL_CACHE_TYPE=3
ENV ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT=1
ENV ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT=5
ENV ATB_OPSRUNNER_KERNEL_CACHE_TILING_SIZE=10240
ENV ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=1
ENV ATB_COMPARE_TILING_EVERY_KERNEL=0
ENV ATB_LAUNCH_KERNEL_WITH_TILING=1
ENV ATB_MATMUL_SHUFFLE_K_ENABLE=1
ENV ATB_RUNNER_POOL_SIZE=64
ENV ASDOPS_MATMUL_PP_FLAG=1
ENV ASDOPS_LOG_LEVEL=ERROR
ENV ASDOPS_LOG_TO_STDOUT=0
ENV ASDOPS_LOG_TO_FILE=1
ENV ASDOPS_LOG_TO_FILE_FLUSH=0
ENV ASDOPS_LOG_TO_BOOST_TYPE=atb
ENV ASDOPS_LOG_PATH=~
ENV ASDOPS_TILING_PARSE_CACHE_DISABLE=0
ENV LCCL_DETERMINISTIC=0

RUN echo "files already in /opt/package" && \
    chmod +x /opt/package/*.run && \
    /opt/package/Ascend-cann-toolkit_*_linux-*.run --install -q && \
    /opt/package/Ascend-cann-${CANN_DEVICE}-ops_${CANN_VERSION}_linux-${ARCH}.run --install -q && \
    /opt/package/Ascend-cann-nnal_*_linux-*.run --install --torch_atb -q && \
    rm -f /opt/package/Ascend-cann-*.run

# ================================================================
# Stage 4: PTA (PyTorch + torch_npu)
# ================================================================
FROM cann AS pta

ARG ARCH
ARG PYTHON_SHORT_VERSION
ARG OS_TYPE

ENV PYTHONPATH=/usr/local/lib/python3.11/site-packages:$PYTHONPATH
ENV PATH=/usr/local/bin:$PATH

# Install PyTorch and torch_npu from downloaded wheels
RUN pip install /opt/package/torch-*.whl && \
    pip install /opt/package/torch_npu-*.whl && \
    rm -f /opt/package/torch-*.whl /opt/package/torch_npu-*.whl

# ================================================================
# Stage 4.5: dev/tools
# ================================================================
FROM pta AS mindstudio

ARG OS_TYPE
ARG ARCH

# system tools
RUN if [ "$OS_TYPE" = "ubuntu" ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            ffmpeg libreadline-dev tree unzip openssl rapidjson-dev libre2-dev \
            libb64-dev libnuma-dev patchelf jq git pkg-config lbzip2 bzip2 gcc \
            make sudo build-essential curl wget cmake vim gdb tcpdump pciutils && \
        rm -rf /var/lib/apt/lists/* ; \
    else \
        dnf update -y && \
        dnf --setopt=sslverify=false install -y ffmpeg readline-devel tree unzip \
            openssl rapidjson-devel re2-devel patchelf jq pkg-config libzip bzip2 \
            sudo curl gdb tcpdump git cmake gcc-c++ && \
        dnf groupinstall "Development Tools" -y ; \
    fi

RUN pip install pybind11 && \
	    pip install decorator && \
	    pip install scipy==1.10.1

# mindstudio tools: msmodelslim / aisbench_benchmark / msprechecker / msprobe
# install in parallel to speed up the build
RUN export TORCH_DEVICE_BACKEND_AUTOLOAD=0 && \
    cd /opt/package && \
    ( \
        git clone https://gitcode.com/Ascend/msmodelslim && \
        cd msmodelslim && bash install.sh && \
        cd /opt/package && rm -rf msmodelslim && \
        echo '*****msmodelslim done*****' \
    ) & \
    ( \
        git clone https://gitee.com/aisbench/benchmark && \
        cd benchmark && pip3 install ./ && \
        pip3 install -r requirements/api.txt && \
        pip3 install -r requirements/extra.txt && \
        cd /opt/package && rm -rf benchmark && \
        echo '*****aisbench_benchmark done*****' \
    ) & \
    ( \
        git clone https://gitcode.com/Ascend/msit && \
        cd msit/msprechecker && pip install . && \
        cd /opt/package && rm -rf msit && \
        echo '*****msprechecker done*****' \
    ) & \
    ( \
        git clone https://gitcode.com/Ascend/msprobe && \
        cd msprobe && pip install . && \
        cd /opt/package && rm -rf msprobe && \
        echo '*****msprobe done*****' \
    ) & \
    wait

# ================================================================
# Stage 5: MindIE
# ================================================================
FROM mindstudio AS mindie

ARG PYTHON_SHORT_VERSION
ARG MINDIE_VERSION
ARG ARCH
ARG PKG_TYPE

# Persist version for agreement banner
ENV MINDIE_VER=${MINDIE_VERSION}

ENV PYTHON_SITE_PACKAGES=/usr/local/lib/python3.11/site-packages

# --------------------------------------------
# mindie-sd
# --------------------------------------------
ENV ASCEND_CUSTOM_OPP_PATH=${PYTHON_SITE_PACKAGES}/mindiesd/ops/vendors/customize:${PYTHON_SITE_PACKAGES}/mindiesd/ops/vendors/aie_ascendc${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}

# --------------------------------------------
# mindie-motor (mindie-service)
# --------------------------------------------
ENV MIES_INSTALL_PATH=${PYTHON_SITE_PACKAGES}/mindie_motor \
    LD_LIBRARY_PATH=${PYTHON_SITE_PACKAGES}/mindie_motor/lib:${PYTHON_SITE_PACKAGES}/mindie_motor/lib/grpc:$LD_LIBRARY_PATH \
    PYTHONPATH=${PYTHON_SITE_PACKAGES}/mindie_motor/bin:$PYTHONPATH

ENV ATB_OPERATION_EXECUTE_ASYNC=1 \
    TASK_QUEUE_ENABLE=1 \
    HCCL_BUFFSIZE=120 \
    MINDIE_LOG_TO_STDOUT=0 \
    MINDIE_LOG_TO_FILE=1 \
    ASCEND_SLOG_PRINT_TO_STDOUT=0 \
    ASCEND_GLOBAL_LOG_LEVEL=3 \
    ASCEND_GLOBAL_EVENT_ENABLE=0 \
    ATB_LOG_TO_FILE_FLUSH=0 \
    ASDOPS_LOG_TO_FILE=0 \
    ASDOPS_LOG_TO_STDOUT=0 \
    ASDOPS_LOG_LEVEL=ERROR

# --------------------------------------------
# mindie-llm
# --------------------------------------------
ENV MINDIE_LLM_HOME_PATH=${PYTHON_SITE_PACKAGES}/mindie_llm \
    LD_LIBRARY_PATH=${PYTHON_SITE_PACKAGES}/mindie_llm/lib:${PYTHON_SITE_PACKAGES}/mindie_llm/lib/grpc:$LD_LIBRARY_PATH \
    PYTHONPATH=${PYTHON_SITE_PACKAGES}/mindie_llm:${PYTHON_SITE_PACKAGES}/mindie_llm/lib:$PYTHONPATH

ENV MINDIE_LLM_CONTINUOUS_BATCHING=1 \
    MINDIE_LLM_RECOMPUTE_THRESHOLD=0.5

# --------------------------------------------
# atb-llm
# --------------------------------------------
ENV ATB_SPEED_HOME_PATH=${PYTHON_SITE_PACKAGES}/atb_llm \
    LD_LIBRARY_PATH=${PYTHON_SITE_PACKAGES}/atb_llm/lib:$LD_LIBRARY_PATH \
    PYTHONPATH=${PYTHON_SITE_PACKAGES}/atb_llm:$PYTHONPATH \
    PYTORCH_INSTALL_PATH=${PYTHON_SITE_PACKAGES}/torch \
    PYTORCH_NPU_INSTALL_PATH=${PYTHON_SITE_PACKAGES}/torch_npu

ENV TASK_QUEUE_ENABLE=1 \
    ATB_OPERATION_EXECUTE_ASYNC=1 \
    ATB_CONTEXT_HOSTTILING_RING=1 \
    ATB_CONTEXT_HOSTTILING_SIZE=102400 \
    ATB_USE_TILING_COPY_STREAM=0 \
    ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT=1 \
    ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT=16

# Install triton-ascend + MindIE components
RUN pip install triton-ascend==3.2.1 \
        --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple \
        --trusted-host triton-ascend.osinfra.cn && \
    if [ "$PKG_TYPE" = "whl" ]; then \
        cd /opt/package && \
        old_umask=$(umask); umask 027; \
        pip install *.whl --no-deps && \
        ln -sf /usr/local/bin/mindie_llm_server /usr/local/bin/mindieservice_daemon && \
        ln -sf /usr/local/bin/mindie_motor_coordinator /usr/local/bin/ms_coordinator && \
        ln -sf /usr/local/bin/mindie_motor_controller /usr/local/bin/ms_controller && \
        umask $old_umask && \
        mkdir -p ${MINDIE_LLM_HOME_PATH}/logs && \
        chmod 750 ${MINDIE_LLM_HOME_PATH}/logs && \
        mkdir -p ${MIES_INSTALL_PATH}/conf && \
        cp -f ${MINDIE_LLM_HOME_PATH}/conf/config.json ${MIES_INSTALL_PATH}/conf/config.json && \
        chmod 550 ${MIES_INSTALL_PATH}/examples/kubernetes_deploy_scripts/boot_helper/* && \
        chmod 640 ${MIES_INSTALL_PATH}/examples/kubernetes_deploy_scripts/boot_helper/boot.sh && \
        chmod 500 ${MIES_INSTALL_PATH}/scripts/http_client_ctl/* && \
        chmod 700 /root && \
        rm -f /opt/package/*.whl; \
    else \
        cd /opt/package && \
        chmod +x *.run && \
        ./*.run --install -q && \
        rm -f /opt/package/*.run; \
    fi && \
    pip cache purge

# Shared object path fallback
ENV LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/lib64:$LD_LIBRARY_PATH

# --------------------------------------------
# Static initialization (users, dirs, perms)
# Common to all supported chips (310 / 910 / A3).
# --------------------------------------------
RUN mkdir -m 750 /var/driver && \
    mkdir -m 750 /var/dmp && \
    mkdir -m 750 /usr/slog && \
    mkdir -m 755 /home/drv && \
    mkdir -m 750 /home/drv/hdc_ppc && \
    mkdir -m 750 -p /var/log/npu/slog && \
    ldconfig && \
    ln -sf /lib /lib64 && \
    userdel -r ubuntu || true && \
    groupadd -g 1000 HwHiAiUser && useradd -u 1000 -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser && \
    groupadd -g 1101 HwDmUser && useradd -u 1101 -g HwDmUser -d /home/HwDmUser -m HwDmUser && \
    groupadd -g 1102 HwBaseUser && useradd -u 1102 -g HwBaseUser -d /home/HwBaseUser -m HwBaseUser && \
    groupadd -g 1100 HwSysUser && useradd -u 1100 -g HwSysUser -d /home/HwSysUser -m HwSysUser && \
    usermod -a -G HwBaseUser HwHiAiUser && \
    usermod -a -G HwDmUser HwHiAiUser && \
    usermod -a -G HwBaseUser HwDmUser && \
    usermod -a -G HwHiAiUser HwDmUser && \
    usermod -a -G HwSysUser HwSysUser && \
    chown HwDmUser:HwDmUser /var/dmp && \
    chown HwHiAiUser:HwHiAiUser /var/driver && \
    chown HwHiAiUser:HwHiAiUser /usr/slog && \
    chown HwHiAiUser:HwHiAiUser /home/drv/hdc_ppc && \
    chown HwHiAiUser:HwHiAiUser /var/log/npu/slog

# Copy entrypoint script and license
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY LICENSE /opt/mindie/LICENSE
RUN chmod +x /usr/local/bin/entrypoint.sh

# Agreement banner (printed by entrypoint on every container start)
WORKDIR /workspace
RUN printf '%s\n' \
    "Version: ${MINDIE_VERSION}" \
    "Container image Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved." \
    "This container image and its contents are governed by the Huawei Container License Agreement (\"License\"). By pulling and using the container, you accept the terms and conditions of this License." \
    "A copy of this License is made available in this container at: https://www.hiascend.com/en/legal/ascendhub-download" \
    "Note: You agree and undertake that when using Huawei or third-party software in this image, you will comply with the license agreement of the corresponding Huawei or third-party software." \
    > /workspace/agreement.txt

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
