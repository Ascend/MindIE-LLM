#!/bin/bash
# ============================================================
# entrypoint.sh — MindIE container entrypoint
# Prints the agreement banner, loads Ascend/CANN/ATB environment
# scripts, then executes the user-supplied command (or bash).
# ============================================================
set -euo pipefail

echo ""
echo "=========================================="
echo "  MindIE image version: ${MINDIE_VER}"
echo "=========================================="
echo ""
cat /opt/mindie/LICENSE
echo ""

load_env_script() {
    local script="$1"
    if [ -f "$script" ]; then
        . "$script"
    fi
}

# Agreement / license banner
cat /workspace/agreement.txt

# Ascend toolkit
load_env_script /usr/local/Ascend/ascend-toolkit/set_env.sh

# CANN IR
shopt -s nullglob
for set_env in /usr/local/Ascend/cann-*/share/info/ascendnpu-ir/bin/set_env.sh; do
    load_env_script "$set_env"
    break
done

# NNAL / ATB
load_env_script /usr/local/Ascend/nnal/atb/set_env.sh

# Execute the user command, default to bash if none provided
exec "${@:-bash}"
