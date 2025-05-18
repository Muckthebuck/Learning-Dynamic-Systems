#!/bin/bash

# --- Usage ---
usage() {
    echo "Usage: bash run_sims.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --config <file>                      YAML config file path (required)"
    echo "  --sim <Pendulum|Cart-Pendulum|Carla|armax>   Simulation type"
    echo "  --T <float>                          Total simulation time"
    echo "  --dt <float>                         Simulation time step"
    echo "  --plot_system / --no-plot_system     Enable/disable plotting"
    echo "  --history_limit <float>             History limit for plotting"
    echo "  --dB <filename>                      Database file"
    echo "  --apply_disturbance                  Apply disturbance"
    echo "  --disturbance <float>               Disturbance force"
    echo "  --controller <str>                  Controller type (default: lqr)"
    echo "  --L <array>                         Reference gain (required)"
    echo "  --reference <list>                  Reference signal type (required)"
    echo "  --r_a <array>                       Reference amplitudes (required)"
    echo "  --r_f <array>                       Reference frequencies (required)"
    echo "  --A <array>                         ARMAX A polynomial"
    echo "  --B <array>                         ARMAX B polynomial"
    echo "  --C <array>                         ARMAX C polynomial"
    echo "  -h, --help                           Show help"
    exit 0
}

CONFIG_FILE=""

# --- Parse --config first ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            shift
            ;;
    esac
done

if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: --config FILE is required."
    usage
fi

# --- Load YAML config ---
if ! command -v yq &>/dev/null; then
    echo "yq is required to load YAML config. Please install it."
    exit 1
fi

SIM_TYPE=$(yq '.sim' "$CONFIG_FILE")
T=$(yq '.T' "$CONFIG_FILE")
dT=$(yq '.dt' "$CONFIG_FILE")
HISTORY_LIMIT=$(yq '.history_limit' "$CONFIG_FILE")
DB_FILE=$(yq '.dB' "$CONFIG_FILE")
DISTURBANCE=$(yq '.disturbance' "$CONFIG_FILE")
CONTROLLER=$(yq '.controller' "$CONFIG_FILE")
L=$(yq -o=json '.L' "$CONFIG_FILE")
REFERENCE=$(yq -o=json '.reference' "$CONFIG_FILE")
R_A=$(yq -o=json '.r_a' "$CONFIG_FILE")
R_F=$(yq -o=json '.r_f' "$CONFIG_FILE")
A=$(yq -o=json '.A // ""' "$CONFIG_FILE")
B=$(yq -o=json '.B // ""' "$CONFIG_FILE")
C=$(yq -o=json '.C // ""' "$CONFIG_FILE")

if [[ $(yq '.plot_system // true' "$CONFIG_FILE") == "false" ]]; then
    PLOT_ARG="--no-plot_system"
else
    PLOT_ARG="--plot_system"
fi

if [[ $(yq '.apply_disturbance // false' "$CONFIG_FILE") == "true" ]]; then
    DIST_ARG="--apply_disturbance"
else
    DIST_ARG=""
fi

# --- Parse CLI overrides ---
set -- "$@"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --sim) SIM_TYPE="$2"; shift 2 ;;
        --T) T="$2"; shift 2 ;;
        --dt) dT="$2"; shift 2 ;;
        --plot_system) PLOT_ARG="--plot_system"; shift ;;
        --no-plot_system) PLOT_ARG="--no-plot_system"; shift ;;
        --history_limit) HISTORY_LIMIT="$2"; shift 2 ;;
        --dB) DB_FILE="$2"; shift 2 ;;
        --apply_disturbance) DIST_ARG="--apply_disturbance"; shift ;;
        --disturbance) DISTURBANCE="$2"; shift 2 ;;
        --controller) CONTROLLER="$2"; shift 2 ;;
        --L) L="$2"; shift 2 ;;
        --reference) REFERENCE="$2"; shift 2 ;;
        --r_a) R_A="$2"; shift 2 ;;
        --r_f) R_F="$2"; shift 2 ;;
        --A) A="$2"; shift 2 ;;
        --B) B="$2"; shift 2 ;;
        --C) C="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
done

# --- Check required args ---
if [[ -z "$L" || -z "$REFERENCE" || -z "$R_A" || -z "$R_F" ]]; then
    echo "Error: --L, --reference, --r_a, and --r_f are required."
    exit 1
fi

# --- Activate venv ---
source .venv/Scripts/activate

# --- Prepare ARMAX polynomial args ---
args=()
if [[ -n "$A" ]]; then args+=(--A "$A"); fi
if [[ -n "$B" ]]; then args+=(--B "$B"); fi
if [[ -n "$C" ]]; then args+=(--C "$C"); fi

# --- Run Python module ---
python -m sims.sims \
    --sim "$SIM_TYPE" \
    --T "$T" \
    --dt "$dT" \
    $PLOT_ARG \
    --history_limit "$HISTORY_LIMIT" \
    --dB "$DB_FILE" \
    $DIST_ARG \
    --disturbance "$DISTURBANCE" \
    --controller "$CONTROLLER" \
    --L "$L" \
    --reference "$REFERENCE" \
    --r_a "$R_A" \
    --r_f "$R_F" \
    "${args[@]}"

# --- Deactivate ---
deactivate
