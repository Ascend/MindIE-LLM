#!/bin/bash
# ============================================================
# config.sh — Central configuration, base URLs, URL templates
# ============================================================
set -euo pipefail

# Guard against re-sourcing (build.sh + download.sh both source this)
if [[ -n "${_CONFIG_SOURCED:-}" ]]; then
    return 0
fi
readonly _CONFIG_SOURCED=1

# Docker registry mirror — used to pre-pull base images
# when direct access to Docker Hub is unavailable
readonly MIRROR_REGISTRY="swr.cn-north-4.myhuaweicloud.com/inference"

# ================================================================
# Logging
# ================================================================

readonly LOG_PREFIX="[MindIE-Docker]"

log_info()  { echo "${LOG_PREFIX} [INFO] $*"; }
log_warn()  { echo "${LOG_PREFIX} [WARN] $*" >&2; }
log_error() { echo "${LOG_PREFIX} [ERROR] $*" >&2; }

# ================================================================
# Base URLs — centralized, single source of truth
# ================================================================

readonly BASE_URL_MINDIE_LLM="https://gitcode.com/Ascend/MindIE-LLM/releases/download"
readonly BASE_URL_MINDIE_SD="https://gitcode.com/Ascend/MindIE-SD/releases/download"
readonly BASE_URL_MINDIE_MOTOR="https://gitcode.com/Ascend/MindIE-Motor/releases/download"
readonly BASE_URL_PTA_RELEASE="https://gitcode.com/Ascend/pytorch/releases/download"
readonly BASE_URL_TORCH="https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch"
readonly BASE_URL_CANN="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN"
readonly BASE_URL_PYTHON_SRC="https://mirrors.huaweicloud.com/python"

# ================================================================
# Validated option sets
# ================================================================

readonly VALID_OS=("ubuntu" "openeuler")
readonly VALID_CHIPS=("310" "910" "A3")
readonly VALID_ARCHS=("x86_64" "aarch64")
readonly VALID_TYPES=("whl" "run")

# ================================================================
# OS metadata
# ================================================================

declare -A OS_BASE_IMAGE=(
    ["ubuntu"]="ubuntu:24.04"
    ["openeuler"]="hub.oepkgs.net/openeuler/openeuler:24.03-lts-sp3"
)

declare -A OS_CODENAME=(
    ["ubuntu"]="ubuntu24.04"
    ["openeuler"]="openeuler24.03-lts"
)

# ================================================================
# Chip metadata
# ================================================================

declare -A CHIP_LABEL=(
    ["310"]="300I-Duo"
    ["910"]="800I-A2"
    ["A3"]="800I-A3"

)

declare -A CANN_DEVICE=(
    ["310"]="310p"
    ["910"]="910b"
    ["A3"]="A3"
)

# ================================================================
# Python helpers
# ================================================================

# Derive cp tag from python version: 3.11.10 → cp311
get_cp_tag() {
    local py_ver="$1"
    local major_minor="${py_ver%.*}"
    echo "cp${major_minor/./}"
}

# Derive python major.minor: 3.11.10 → 3.11
get_py_major() {
    local py_ver="$1"
    echo "${py_ver%.*}"
}

# Derive python short: 3.11.10 → 311
get_py_short() {
    local py_ver="$1"
    local major_minor="${py_ver%.*}"
    echo "${major_minor/./}"
}

# ================================================================
# URL templates — use BASE_URL_* constants
# ================================================================

# mindie-llm .whl
# pattern: mindie_llm-{ver}-{cp_tag}-{cp_tag}-linux_{arch}.whl
url_mindie_llm_whl() {
    local ver="$1" arch="$2" cp_tag="$3"
    echo "${BASE_URL_MINDIE_LLM}/${ver}/mindie_llm-${ver}-${cp_tag}-${cp_tag}-linux_${arch}.whl"
}

# mindie-llm .run package
url_mindie_llm_run() {
    local ver="$1" arch="$2"
    echo "${BASE_URL_MINDIE_LLM}/${ver}/Ascend-mindie_${ver}_linux-${arch}_abi1.run"
}

# atb-llm .whl
url_atb_llm_whl() {
    local ver="$1" arch="$2" cp_tag="$3"
    echo "${BASE_URL_MINDIE_LLM}/${ver}/atb_llm-${ver}-${cp_tag}-${cp_tag}-linux_${arch}.whl"
}

# mindie-sd .whl
# pattern: mindiesd-{ver}-{cp_tag}-{cp_tag}-linux_{arch}.whl
url_mindie_sd_whl() {
    local ver="$1" arch="$2" cp_tag="$3"
    echo "${BASE_URL_MINDIE_SD}/${ver}/mindiesd-${ver}-${cp_tag}-${cp_tag}-linux_${arch}.whl"
}

# mindie-motor .whl
# pattern: mindie_motor-{ver}-{cp_tag}-{cp_tag}-linux_{arch}.whl
url_mindie_motor_whl() {
    local ver="$1" arch="$2" cp_tag="$3"
    echo "${BASE_URL_MINDIE_MOTOR}/${ver}/mindie_motor-${ver}-${cp_tag}-${cp_tag}-linux_${arch}.whl"
}

url_pta_whl() {
    local pta_tag="$1" cp_tag="$2" arch="$3"

    local torch_ver base_url ver url status
    torch_ver=$(sed -n 's/.*pytorch\([0-9.]*\).*/\1/p' <<<"$pta_tag")
    base_url="${BASE_URL_PTA_RELEASE}/${pta_tag}"

    # Find torch_npu wheel
    url="${base_url}/torch_npu-${torch_ver}-${cp_tag}-${cp_tag}-manylinux_2_28_${arch}.whl"
    status=$(curl -k -L -s -o /dev/null -w "%{http_code}" -A "Mozilla/5.0" "$url")
    if [[ "$status" == "200" || "$status" == "302" ]]; then
        echo "$url"
        return 0
    fi

    # post releases
    for i in {1..10}; do
        ver="${torch_ver}.post${i}"
        url="${base_url}/torch_npu-${ver}-${cp_tag}-${cp_tag}-manylinux_2_28_${arch}.whl"

        status=$(curl -k -L -s -o /dev/null -w "%{http_code}" -A "Mozilla/5.0" "$url")
        if [[ "$status" == "200" || "$status" == "302" ]]; then
            echo "$url"
            return 0
        fi
    done

    # rc releases
    for i in {1..5}; do
        ver="${torch_ver}rc${i}"
        url="${base_url}/torch_npu-${ver}-${cp_tag}-${cp_tag}-manylinux_2_28_${arch}.whl"

        status=$(curl -k -L -s -o /dev/null -w "%{http_code}" -A "Mozilla/5.0" "$url")
        if [[ "$status" == "200" || "$status" == "302" ]]; then
            echo "$url"
            return 0
        fi
    done

    log_error "could not find torch_npu wheel"
    log_error "pta_tag=${pta_tag}"
    log_error "cp_tag=${cp_tag}"
    log_error "arch=${arch}"
    return 1
}

# Find torch wheel URL from OBS.
# x86_64 uses "%2Bcpu" suffix between version and cp_tag; aarch64 does not.
# Returns the URL on stdout, non-zero exit on failure.
url_torch_whl() {
    local pta_tag="$1" cp_tag="$2" arch="$3"

    local torch_ver url status suffix
    torch_ver=$(sed -n 's/.*pytorch\([0-9.]*\).*/\1/p' <<<"$pta_tag")

    if [[ "$arch" == "x86_64" ]]; then
        suffix="%2Bcpu"
    else
        suffix=""
    fi

    url="${BASE_URL_TORCH}/v${torch_ver}/torch-${torch_ver}${suffix}-${cp_tag}-${cp_tag}-manylinux_2_28_${arch}.whl"
    status=$(curl -k -L -s -o /dev/null -w "%{http_code}" -A "Mozilla/5.0" "$url")
    if [[ "$status" == "200" || "$status" == "302" ]]; then
        echo "$url"
        return 0
    fi

    log_error "could not find torch wheel"
    log_error "url=${url}"
    log_error "status=${status}"
    return 1
}

# CANN toolkit
url_cann_toolkit() {
    local cann_ver="$1" arch="$2"
    echo "${BASE_URL_CANN}%20${cann_ver}/Ascend-cann-toolkit_${cann_ver}_linux-${arch}.run"
}

# CANN nnal
url_cann_nnal() {
    local cann_ver="$1" arch="$2"
    echo "${BASE_URL_CANN}%20${cann_ver}/Ascend-cann-nnal_${cann_ver}_linux-${arch}.run"
}

# CANN kernels (chip-specific ops)
url_cann_kernels() {
    local cann_ver="$1" arch="$2" chip="$3"
    local device="${CANN_DEVICE[$chip]}"
    echo "${BASE_URL_CANN}%20${cann_ver}/Ascend-cann-${device}-ops_${cann_ver}_linux-${arch}.run"
}

# Python source tarball
url_python_src() {
    local py_ver="$1"
    echo "${BASE_URL_PYTHON_SRC}/${py_ver}/Python-${py_ver}.tar.xz"
}

# ================================================================
# arch detection
# ================================================================

arch_detect() {
    local a
    a=$(uname -m)
    case "$a" in
        x86_64)    echo "x86_64" ;;
        aarch64)   echo "aarch64" ;;
        *)
            log_error "unsupported arch: $a"
            return 1
            ;;
    esac
}

# ================================================================
# validation
# ================================================================

validate_config() {
    local os="$1" chip="$2" type_="$3" arch="${4:-}"
    local errors=0

    if [[ ! " ${VALID_OS[*]} " =~ " ${os} " ]]; then
        log_error "invalid os '${os}'. Must be one of: ${VALID_OS[*]}"
        errors=$((errors + 1))
    fi

    if [[ ! " ${VALID_CHIPS[*]} " =~ " ${chip} " ]]; then
        log_error "invalid chip '${chip}'. Must be one of: ${VALID_CHIPS[*]}"
        errors=$((errors + 1))
    fi

    if [[ ! " ${VALID_ARCHS[*]} " =~ " ${arch} " ]]; then
        log_error "invalid arch '${arch}'. Must be one of: ${VALID_ARCHS[*]}"
        errors=$((errors + 1))
    fi

    if [[ ! " ${VALID_TYPES[*]} " =~ " ${type_} " ]]; then
        log_error "invalid type '${type_}'. Must be one of: ${VALID_TYPES[*]}"
        errors=$((errors + 1))
    fi

    return $errors
}
