#!/bin/bash
# ============================================================
# entrypoint.sh — MindIE-LLM container entrypoint
# Prints version + license banner on every container start,
# then executes the user-supplied command (or bash by default).
# ============================================================
set -euo pipefail

echo ""
echo "=========================================="
echo "  MindIE-LLM image version: ${MINDIE_LLM_VER}"
echo "=========================================="
echo ""
cat /opt/mindie/LICENSE
echo ""

# Execute the user command, default to bash if none provided
exec "${@:-bash}"
