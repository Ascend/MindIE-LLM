#!/bin/bash

ROOT_DIR=$1
SOC_VERSION=$2

if [[ "$SOC_VERSION" =~ ^ascend910b ]]; then
    # ASCEND910B (A2) series
    # dependency: catlass
    CATLASS_PATH=${ROOT_DIR}/../../../third_party/catlass/include
    ABSOLUTE_CATLASS_PATH=$(cd "${CATLASS_PATH}" && pwd)
    export CPATH=${ABSOLUTE_CATLASS_PATH}:${CPATH}

    CUSTOM_OPS_ARRAY=(
        "lightning_indexer"
    )

    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend910b"
elif [[ "$SOC_VERSION" =~ ^ascend910_93 ]]; then
    # ASCEND910C (A3) series
    # dependency: catlass
    CATLASS_PATH=${ROOT_DIR}/../../../third_party/catlass/include

    CUSTOM_OPS_ARRAY=(
        "dispatch_ffn_combine"
        "dispatch_gmm_combine_decode"
    )
    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend910_93"
else
    # others
    # currently, no custom aclnn ops for other series
    exit 0
fi


(
  set -euo pipefail

  cd csrc

  : "${ROOT_DIR:?ROOT_DIR is not set}"
  : "${CUSTOM_OPS:?CUSTOM_OPS is not set}"
  : "${SOC_VERSION:?SOC_VERSION is not set}"
  : "${SOC_ARG:?SOC_ARG is not set}"

  echo "building custom ops ${CUSTOM_OPS} for ${SOC_VERSION}"
  bash build.sh --pkg --ops="${CUSTOM_OPS}" --soc="${SOC_ARG}"

  shopt -s nullglob
  runs=(./build/cann-ops-transformer*.run)
  shopt -u nullglob

  (( ${#runs[@]} == 1 )) || { echo "ERROR: expected 1 installer, got ${#runs[@]}" >&2; exit 1; }

  chmod +x -- "${runs[0]}" || true
)
