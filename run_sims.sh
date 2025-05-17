#!/bin/bash

# Default values
SIM_TYPE="Pendulum"
T=10
dT=0.02
PLOT_ARG="--plot_system"
HISTORY_LIMIT=2.0
DB_FILE="sim_data.db"
DIST_ARG=""
DISTURBANCE=50
CONTROLLER="lqr"
L=""
REFERENCE=""
R_A=""
R_F=""
A=""
B=""
C=""
CONFIG_FILE=""

# --- Usage ---
usage() {
    echo "Usage: bash run_sims.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --config <file>                      YAML config file path"
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

# --- Load YAML config if provided ---
if [[ -n "$CONFIG_FILE" ]]; then
    if ! command -v yq &>/dev/null; then
        echo "yq is required to load YAML config. Please install it."
        exit 1
    fi

    SIM_TYPE=$(yq '.sim // "Pendulum"' "$CONFIG_FILE")
    T=$(yq '.T // 10' "$CONFIG_FILE")
    dT=$(yq '.dt // 0.02' "$CONFIG_FILE")
    HISTORY_LIMIT=$(yq '.history_limit // 2.0' "$CONFIG_FILE")
    DB_FILE=$(yq '.dB // "sim_data.db"' "$CONFIG_FILE")
    DISTURBANCE=$(yq '.disturbance // 50' "$CONFIG_FILE")
    CONTROLLER=$(yq '.controller // "lqr"' "$CONFIG_FILE")
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
    fi
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
    $( [[ -n "$A" ]] && echo "--A $A" ) \
    $( [[ -n "$B" ]] && echo "--B $B" ) \
    $( [[ -n "$C" ]] && echo "--C $C" )

# --- Deactivate ---
deactivate
