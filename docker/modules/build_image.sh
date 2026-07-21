#!/bin/bash
# ============================================================
# build_image.sh — Docker build orchestration
# ============================================================
set -euo pipefail

source "${MODULES_DIR}/config.sh"

readonly DOCKERFILE="${DOCKER_DIR}/Dockerfile"

# ---------- tag computation ----------

_image_tag() {
    local os="$1" chip="$2" arch="$3" mindie_ver="$4"
    local os_code="${OS_CODENAME[$os]}"
    local chip_label="${CHIP_LABEL[$chip]}"
    echo "mindie:${mindie_ver}-${chip_label}-py3.11-${os_code}-${arch}"
}

# ---------- cleanup ----------

docker_cleanup() {
    set +e
    local containers
    containers=$(docker ps -a -q -f status=exited 2>/dev/null)
    if [[ -n "$containers" ]]; then
        docker rm "$containers" 2>/dev/null || true
    fi

    local dangling
    dangling=$(docker images -f "dangling=true" -q 2>/dev/null)
    if [[ -n "$dangling" ]]; then
        docker rmi -f "$dangling" 2>/dev/null || true
    fi
}

# ---------- base image pre-pull (via mirror if needed) ----------

_prepull_base_image() {
    local os="$1"
    local base_image="${OS_BASE_IMAGE[$os]}"

    # Already cached locally — skip
    if docker image inspect "$base_image" &>/dev/null; then
        log_info "base image already cached: $base_image"
        return 0
    fi

    # Try mirror first, fall back to direct pull if mirror is unavailable
    local mirror_image="${MIRROR_REGISTRY}/${base_image}"
    log_info "pulling base image from mirror: $mirror_image"
    if docker pull "$mirror_image" 2>/dev/null; then
        docker tag "$mirror_image" "$base_image"
        log_info "base image ready (via mirror): $base_image"
        return 0
    fi

    log_warn "mirror pull failed, falling back to direct pull: $base_image"
    if docker pull "$base_image"; then
        log_info "base image ready (direct): $base_image"
        return 0
    fi

    log_error "failed to pull base image: $base_image"
    return 1
}

# ---------- main build ----------

run_build() {
    local os="$1" chip="$2" arch="$3" type="$4"
    local cann_ver="$5" mindie_ver="$6" atb_llm_ver="$7"
    local pta_tag="$8" python_ver="$9" mindie_sd_ver="${10}" mindie_motor_ver="${11}"

    local py_short py_major cann_device image_tag
    py_short=$(get_py_short "$python_ver")
    py_major=$(get_py_major "$python_ver")
    cann_device="${CANN_DEVICE[$chip]}"
    image_tag=$(_image_tag "$os" "$chip" "$arch" "$mindie_ver")

    log_info "========== Build START =========="
    log_info "image tag: ${image_tag}"

    docker_cleanup

    _prepull_base_image "$os" || {
        log_error "base image pull failed, aborting build"
        return 1
    }

    log_info "building base + mindie image (single pass, multi-stage)..."
    DOCKER_BUILDKIT=1 docker build \
        --build-arg OS_TYPE="$os" \
        --build-arg CHIP="$chip" \
        --build-arg ARCH="$arch" \
        --build-arg PKG_TYPE="$type" \
        --build-arg PYTHON_VERSION="$python_ver" \
        --build-arg PYTHON_MAJOR_VERSION="$py_major" \
        --build-arg PYTHON_SHORT_VERSION="$py_short" \
        --build-arg CANN_VERSION="$cann_ver" \
        --build-arg CANN_DEVICE="$cann_device" \
        --build-arg MINDIE_VERSION="$mindie_ver" \
        --network=host \
        -t "$image_tag" \
        -f "$DOCKERFILE" \
        "$DOCKER_DIR"

    docker images "$image_tag"

    save_image "$image_tag" "$os" "$chip" "$arch"
    docker_cleanup

    log_info "========== Build DONE: ${image_tag} =========="
}

save_image() {
    local image_tag="$1" os="$2" chip="$3" arch="$4"

    local output_file="${OUTPUT_DIR}/${image_tag}.tar.gz"
    mkdir -p "$OUTPUT_DIR"

    log_info "exporting ${image_tag} -> ${output_file}"
    docker save "$image_tag" | pigz -c > "$output_file"
    log_info "image saved: ${output_file}"
}
