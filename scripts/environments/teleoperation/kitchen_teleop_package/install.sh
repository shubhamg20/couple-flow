#!/usr/bin/env bash
set -euo pipefail

# Kitchen Teleop Installer
# Usage: Place this folder anywhere inside your IsaacLab repo, then run ./install.sh
# It finds ISAACLAB_ROOT by walking up from the script location looking for isaaclab.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Find ISAACLAB_ROOT ---
if [ -n "${ISAACLAB_ROOT:-}" ]; then
    echo "Using ISAACLAB_ROOT=$ISAACLAB_ROOT"
else
    DIR="$SCRIPT_DIR"
    while [ "$DIR" != "/" ]; do
        if [ -f "$DIR/isaaclab.sh" ]; then
            ISAACLAB_ROOT="$DIR"
            break
        fi
        DIR="$(dirname "$DIR")"
    done
    if [ -z "${ISAACLAB_ROOT:-}" ]; then
        echo "ERROR: Could not find isaaclab.sh in any parent directory."
        echo "Set ISAACLAB_ROOT manually: ISAACLAB_ROOT=/path/to/IsaacLab ./install.sh"
        exit 1
    fi
    echo "Found ISAACLAB_ROOT=$ISAACLAB_ROOT"
fi

PICK_PLACE="$ISAACLAB_ROOT/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place"
MDP_DIR="$PICK_PLACE/mdp"
TELEOP_DIR="$ISAACLAB_ROOT/scripts/environments/teleoperation"
USD_DIR="$ISAACLAB_ROOT/usd_extracted"
KITCHEN_DIR="$ISAACLAB_ROOT/source/kitchen_assets"

echo ""

# 1. Env configs
echo "[1/5] Env configs -> $PICK_PLACE/"
cp "$SCRIPT_DIR/kitchen_franka_env_cfg.py" "$PICK_PLACE/"
cp "$SCRIPT_DIR/kitchen_gr1t2_env_cfg.py" "$PICK_PLACE/"

# 2. MDP files
echo "[2/5] MDP modules -> $MDP_DIR/"
cp "$SCRIPT_DIR/mdp_init.py" "$MDP_DIR/__init__.py"
cp "$SCRIPT_DIR/observations.py" "$MDP_DIR/"
cp "$SCRIPT_DIR/terminations.py" "$MDP_DIR/"
cp "$SCRIPT_DIR/pick_place_events.py" "$MDP_DIR/"

# 3. Append kitchen registrations to __init__.py if not already there
echo "[3/5] Registering kitchen tasks..."
if grep -q "Isaac-Kitchen-Franka-v0" "$PICK_PLACE/__init__.py" 2>/dev/null; then
    echo "  Already registered, skipping."
else
    cat >> "$PICK_PLACE/__init__.py" << 'PYEOF'

# --- Kitchen teleop tasks (appended by install.sh) ---
from . import kitchen_franka_env_cfg, kitchen_gr1t2_env_cfg

gym.register(
    id="Isaac-Kitchen-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": kitchen_franka_env_cfg.KitchenFrankaEnvCfg},
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Kitchen-GR1T2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": kitchen_gr1t2_env_cfg.KitchenGR1T2EnvCfg},
    disable_env_checker=True,
)
PYEOF
    echo "  Appended to $PICK_PLACE/__init__.py"
fi

# 4. Teleop scripts
echo "[4/5] Teleop scripts -> $TELEOP_DIR/"
cp "$SCRIPT_DIR/teleop.py" "$TELEOP_DIR/"
cp "$SCRIPT_DIR/teleop_gr1.py" "$TELEOP_DIR/"

# 5. USD assets (skip dirs that already exist)
echo "[5/5] USD assets..."
for d in bowl_csm mug_csm rack_csm; do
    if [ -d "$USD_DIR/$d" ]; then
        echo "  SKIP $d/ (already exists)"
    else
        cp -r "$SCRIPT_DIR/usd/$d" "$USD_DIR/"
        echo "  -> $USD_DIR/$d/"
    fi
done
mkdir -p "$KITCHEN_DIR"
if [ -f "$KITCHEN_DIR/kitchen_background.usd" ]; then
    echo "  SKIP kitchen_assets/ (already exists)"
else
    cp -r "$SCRIPT_DIR/usd/kitchen_assets/"* "$KITCHEN_DIR/"
    echo "  -> $KITCHEN_DIR/"
fi

echo ""
echo "Done! Tasks registered: Isaac-Kitchen-Franka-v0, Isaac-Kitchen-GR1T2-v0"
