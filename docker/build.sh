#!/bin/bash
# ============================================================
# build.sh — MindIE-LLM Docker image build
# ============================================================
# Usage:
#   ${SCRIPT_REL_PATH} --os=<os> --chip=<chip> --arch=<arch>          \
#              --mindie-llm=<ver>                                  \
#              --cann=<ver> --pta-tag=<tag>                      \
#              [--type=<whl|run>] [--python=<ver>]               \
#              [--dry-run]
#
# Examples:
#   ${SCRIPT_REL_PATH} --os=ubuntu --chip=910 --arch=x86_64          \
#              --mindie-llm=3.0.0                                  \
#              --cann=8.2.RC1 --pta-tag=v26.0.0-pytorch2.7.1    \
#              --type=run --python=3.11.6
#
# ============================================================

set -euo pipefail

DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REL_PATH="$(realpath --relative-to="$PWD" "${BASH_SOURCE[0]}")"
MODULES_DIR="${DOCKER_DIR}/modules"

source "${MODULES_DIR}/config.sh"

# ---------- defaults ----------
OUTPUT_DIR="${DOCKER_DIR}/output"
DOWNLOAD_DIR="${DOCKER_DIR}/downloads"

# ---------- usage ----------
usage() {
    cat <<EOF
Usage: ${SCRIPT_REL_PATH} [OPTIONS]

Required:
  --os=OS                OS: ubuntu | openeuler
  --chip=CHIP            Chip: 310 | 910 | A3
  --arch=ARCH            Target arch: x86_64 | aarch64
  --mindie-llm=VER       mindie-llm version (e.g. 3.0.0)
  --cann=VER             CANN version (e.g. 9.0.0)
  --pta-tag=TAG          PTA release tag (e.g. v26.0.0-pytorch2.9.0)

Optional:
  --type=TYPE            Package type: whl | run          [default: whl]
  --python=VER           Python version (e.g. 3.11.10)    [default: 3.11.10]
  --dry-run              Validate and show config only, don't build
  -h, --help             Show this help

Examples:
  ${SCRIPT_REL_PATH} --os=ubuntu --chip=910 --arch=x86_64 --mindie-llm=3.0.0 --cann=9.0.0 --pta-tag=v26.0.0-pytorch2.9.0

  ${SCRIPT_REL_PATH} --os=openeuler --chip=310 --arch=aarch64 --mindie-llm=3.0.0 --cann=9.0.0 --pta-tag=v26.0.0-pytorch2.9.0 \\
                --type=run --python=3.11.6
EOF
    exit 0
}

# ---------- argument parsing ----------

parse_args() {
    OS=""
    CHIP=""
    ARCH=""
    TYPE="whl"
    MINDIE_LLM_VER=""
    CANN_VER=""
    PTA_TAG=""
    PYTHON_VER="3.11.10"
    DRY_RUN=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --os=*)              OS="${1#*=}" ;;
            --chip=*)            CHIP="${1#*=}" ;;
            --arch=*)            ARCH="${1#*=}" ;;
            --type=*)            TYPE="${1#*=}" ;;
            --mindie-llm=*)      MINDIE_LLM_VER="${1#*=}" ;;
            --cann=*)            CANN_VER="${1#*=}" ;;
            --pta-tag=*)         PTA_TAG="${1#*=}" ;;
            --python=*)          PYTHON_VER="${1#*=}" ;;
            --dry-run)           DRY_RUN=true ;;
            -h|--help)           usage ;;
            *)
                log_error "unknown option: $1"
                usage ;;
        esac
        shift
    done
}

# ---------- validation ----------

validate_required() {
    local errors=0

    check() {
        local val="$1" flag="$2"
        if [[ -z "$val" ]]; then
            log_error "--${flag} is required"
            errors=$((errors + 1))
        fi
    }

    check "$OS"              "os"
    check "$CHIP"            "chip"
    check "$ARCH"            "arch"
    check "$MINDIE_LLM_VER"  "mindie-llm"
    check "$CANN_VER"        "cann"
    check "$PTA_TAG"         "pta-tag"

    if [[ $errors -gt 0 ]]; then
        echo ""
        log_error "Run with --help for usage information."
        exit 1
    fi

    validate_config "$OS" "$CHIP" "$TYPE" "$ARCH"
}

# ---------- show config ----------

show_config() {
    echo "=========================================="
    echo "  MindIE-LLM Docker Build Configuration"
    echo "=========================================="
    echo "  OS:           ${OS} (${OS_CODENAME[$OS]})"
    echo "  Arch:         ${ARCH}"
    echo "  Chip:         ${CHIP} (${CHIP_LABEL[$CHIP]})"
    echo "  Type:         ${TYPE}"
    echo "  mindie-llm:   ${MINDIE_LLM_VER}"
    echo "  CANN:         ${CANN_VER}"
    echo "  pta-tag:      ${PTA_TAG}"
    echo "  Python:       ${PYTHON_VER}"
    echo "  Downloads:    ${DOWNLOAD_DIR}"
    echo "=========================================="
}

# ---------- main ----------

main() {
    parse_args "$@"
    validate_required

    show_config

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "dry-run passed, skipping build."
        exit 0
    fi

    # Step 1: Download
    source "${MODULES_DIR}/download.sh"
    download_all "$OS" "$CHIP" "$ARCH" "$TYPE" "$CANN_VER" "$MINDIE_LLM_VER" "$MINDIE_LLM_VER" "$PTA_TAG" "$PYTHON_VER"

    # Step 2: Build
    source "${MODULES_DIR}/build_image.sh"
    run_build "$OS" "$CHIP" "$ARCH" "$TYPE" "$CANN_VER" "$MINDIE_LLM_VER" "$MINDIE_LLM_VER" "$PTA_TAG" "$PYTHON_VER"

    # Step 3: License notice
    echo ""
    echo "=========================================="
    echo "  MindIE-LLM image version: ${MINDIE_LLM_VER}"
    echo "=========================================="
    echo ""
    cat "${DOCKER_DIR}/LICENSE"
    echo ""
}

main "$@"
