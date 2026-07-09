#!/bin/bash
# ============================================================
# download.sh — Download layer
# ============================================================
set -euo pipefail

source "${MODULES_DIR}/config.sh"

_download_file() {
    local url="$1" output_name="$2"
    local output_path="${DOWNLOAD_DIR}/${output_name}"

    log_info "url:         ${url}"
    log_info "downloading: ${output_path}"

    # Download with explicit error handling: --tries=5, show progress on failure
    if ! wget -q --tries=5 --timeout=5 \
        --header="Referer: https://www.hiascend.com/" \
        "$url" -O "$output_path" --no-check-certificate; then
        log_error "download failed: ${url}"
        return 1
    fi

    if head -c 512 "$output_path" | grep -qiE '<html|<!doctype html|<head|<body'; then
        log_error "downloaded file appears to be an HTML error page (not a valid package)"
        log_error "url:         ${url}"
        log_error "hint: the file may not exist at this OBS/gitcode path, check the version"
        rm -f "$output_path"
        return 1
    fi

    local fsize
    fsize=$(stat -c%s "$output_path" 2>/dev/null || echo 0)
    log_info "done:        ${output_name} (${fsize} bytes)"
}

# ---------- per-component download ----------

download_python_src() {
    local python_ver="$1" os="$2"

    # openEuler has python pre-installed via dnf
    if [[ "$os" == "openeuler" ]]; then
        log_info "python is pre-installed in openEuler, skip source download"
        return 0
    fi

    local url
    url=$(url_python_src "$python_ver")
    _download_file "$url" "Python-${python_ver}.tar.xz"
}

download_cann_toolkit() {
    local cann_ver="$1" arch="$2"

    local toolkit_url
    toolkit_url=$(url_cann_toolkit "$cann_ver" "$arch")
    _download_file "$toolkit_url" "Ascend-cann-toolkit_${cann_ver}_linux-${arch}.run"
}

download_cann_nnal() {
    local cann_ver="$1" arch="$2"

    local nnal_url
    nnal_url=$(url_cann_nnal "$cann_ver" "$arch")
    _download_file "$nnal_url" "Ascend-cann-nnal_${cann_ver}_linux-${arch}.run"
}

download_cann_kernels() {
    local cann_ver="$1" arch="$2" chip="$3"
    local cann_device="${CANN_DEVICE[$chip]}"

    local kernel_url
    kernel_url=$(url_cann_kernels "$cann_ver" "$arch" "$chip")
    _download_file "$kernel_url" "Ascend-cann-${cann_device}-ops_${cann_ver}_linux-${arch}.run"
}

download_pta() {
    local pta_tag="$1" arch="$2" cp_tag="$3"

    # Download torch_npu wheel
    local npu_url npu_file
    npu_url=$(url_pta_whl "$pta_tag" "$cp_tag" "$arch") || {
        log_error "failed to locate torch_npu wheel"
        return 1
    }
    npu_file=$(basename "$npu_url")
    _download_file "$npu_url" "$npu_file"

    # Download torch wheel
    local torch_url torch_file
    torch_url=$(url_torch_whl "$pta_tag" "$cp_tag" "$arch") || {
        log_error "failed to locate torch wheel"
        return 1
    }
    torch_file=$(basename "$torch_url")
    _download_file "$torch_url" "$torch_file"
}

download_mindie_llm() {
    local mindie_llm_ver="$1" atb_llm_ver="$2" arch="$3" cp_tag="$4" type_="$5"

    if [[ "$type_" == "whl" ]]; then
        local mindie_url
        mindie_url=$(url_mindie_llm_whl "$mindie_llm_ver" "$arch" "$cp_tag")
        _download_file "$mindie_url" "mindie_llm-${mindie_llm_ver}-${cp_tag}-${cp_tag}-linux_${arch}.whl"

        local atb_url
        atb_url=$(url_atb_llm_whl "$atb_llm_ver" "$arch" "$cp_tag")
        _download_file "$atb_url" "atb_llm-${atb_llm_ver}-${cp_tag}-${cp_tag}-linux_${arch}.whl"
    elif [[ "$type_" == "run" ]]; then
        local run_url
        run_url=$(url_mindie_llm_run "$mindie_llm_ver" "$arch")
        _download_file "$run_url" "Ascend-mindie_${mindie_llm_ver}_linux-${arch}_abi1.run"
    fi
}

download_mindie_sd() {
    local mindie_sd_ver="$1" arch="$2" cp_tag="$3"
    local url
    url=$(url_mindie_sd_whl "$mindie_sd_ver" "$arch" "$cp_tag")
    _download_file "$url" "mindiesd-${mindie_sd_ver}-${cp_tag}-${cp_tag}-linux_${arch}.whl"
}

download_mindie_motor() {
    local mindie_motor_ver="$1" arch="$2" cp_tag="$3"
    local url
    url=$(url_mindie_motor_whl "$mindie_motor_ver" "$arch" "$cp_tag")
    _download_file "$url" "mindie_motor-${mindie_motor_ver}-${cp_tag}-${cp_tag}-linux_${arch}.whl"
}

# ---------- orchestration ----------

download_all() {
    local os="$1" chip="$2" arch="$3" type_="$4"
    local cann_ver="$5" mindie_llm_ver="$6" atb_llm_ver="$7"
    local pta_tag="$8" python_ver="$9" mindie_sd_ver="${10}" mindie_motor_ver="${11}"

    local cp_tag
    cp_tag=$(get_cp_tag "$python_ver")

    mkdir -p "$DOWNLOAD_DIR"

    # 8 parallel downloads maximum (sd/motor are no-ops for run type)
    local num_downloads=8
    log_info "========== Download START (${num_downloads} parallel) =========="

    # Run all downloads in parallel, collect exit codes via temp files
    local tmpdir
    tmpdir=$(mktemp -d)
    local pids=()

    # 1. pta whl
    (download_pta "$pta_tag" "$arch" "$cp_tag"; echo $? >"$tmpdir/rc_pta") & pids+=($!)

    # 2. python source
    (download_python_src "$python_ver" "$os"; echo $? >"$tmpdir/rc_python") & pids+=($!)

    # 3. cann toolkit
    (download_cann_toolkit "$cann_ver" "$arch"; echo $? >"$tmpdir/rc_cann_toolkit") & pids+=($!)

    # 4. cann nnal
    (download_cann_nnal "$cann_ver" "$arch"; echo $? >"$tmpdir/rc_cann_nnal") & pids+=($!)

    # 5. cann kernels
    (download_cann_kernels "$cann_ver" "$arch" "$chip"; echo $? >"$tmpdir/rc_cann_kernels") & pids+=($!)

    # 6. mindie llm
    (download_mindie_llm "$mindie_llm_ver" "$atb_llm_ver" "$arch" "$cp_tag" "$type_"; echo $? >"$tmpdir/rc_mindie_llm") & pids+=($!)

    # 7. mindie sd
    (download_mindie_sd "$mindie_sd_ver" "$arch" "$cp_tag"; echo $? >"$tmpdir/rc_mindie_sd") & pids+=($!)

    # 8. mindie motor
    (download_mindie_motor "$mindie_motor_ver" "$arch" "$cp_tag"; echo $? >"$tmpdir/rc_mindie_motor") & pids+=($!)

    # Wait for all background jobs, return first non-zero exit code
    local overall_rc=0
    for pid in "${pids[@]}"; do
        wait "$pid" || true
    done

    # Check each exit code
    local names=("pta" "python_src" "cann_toolkit" "cann_nnal" "cann_kernels" "mindie_llm" "mindie_sd" "mindie_motor")
    local labels=("download_pta" "download_python_src" "download_cann_toolkit" "download_cann_nnal" "download_cann_kernels" "download_mindie_llm" "download_mindie_sd" "download_mindie_motor")
    for i in "${!names[@]}"; do
        local rc_file="$tmpdir/rc_${names[$i]}"
        local rc=0
        [[ -f "$rc_file" ]] && rc=$(cat "$rc_file")
        if [[ "$rc" -ne 0 ]]; then
            log_error "${labels[$i]} failed with exit code $rc"
            overall_rc=1
        fi
    done

    rm -rf "$tmpdir"
    if [[ "$overall_rc" -ne 0 ]]; then
        log_error "one or more downloads failed"
        return 1
    fi

    log_info "========== Download DONE =========="
    log_info "files in ${DOWNLOAD_DIR}:"
    ls -la "$DOWNLOAD_DIR"
}
