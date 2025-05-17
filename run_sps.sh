#!/bin/bash

# Default values
n_states=2
n_inputs=1
n_outputs=1
n_output=1  # Keeping for backward compatibility if used elsewhere
n_A=2
n_B=2
C="[[0, 1]]"
n_noise_order=3
n_points="[11, 21, 11]"
m=100
q=5
N=50
db="sim.db"
debug=false
epsilon=1e-10
LAMBDA="[[1, 0], [0, 1]]"
CONFIG_FILE=""
search=""
bounds=""

# Usage info
usage() {
    echo "Usage: $0 [--config file] [options]"
    echo ""
    echo "Options:"
    echo "  --config FILE          YAML config file"
    echo "  --Lambda LAMBDA        Lambda weight matrix"
    echo "  --n_A N                Polynomial A order"
    echo "  --n_B N                Polynomial B order"
    echo "  --n_states N           Number of states"
    echo "  --n_inputs N           Number of inputs"
    echo "  --n_outputs N          Number of outputs"
    echo "  --C C_MATRIX           C matrix"
    echo "  --n_noise_order N      Noise order"
    echo "  --n_points N           Point grid list"
    echo "  --m M                  Number of samples"
    echo "  --q Q                  Number of query points"
    echo "  --N N                  Number of final samples"
    echo "  --dB DB_FILE           Database file"
    echo "  --debug                Enable debug"
    echo "  --epsilon EPSILON      Epsilon value"
    echo "  --search STRATEGY      Search strategy (grid or radial)"
    echo "  --bounds BOUNDS        Bounds as list-like string"
    echo "  -h                     Show help"
    exit 1
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

# --- Load config from YAML if provided ---
if [[ -n "$CONFIG_FILE" ]]; then
    if ! command -v yq &> /dev/null; then
        echo "Error: 'yq' not found. Install it to use YAML config support."
        exit 1
    fi

    n_states=$(yq '.n_states // 2' "$CONFIG_FILE")
    n_inputs=$(yq '.n_inputs // 1' "$CONFIG_FILE")
    n_outputs=$(yq '.n_outputs // 1' "$CONFIG_FILE")
    n_A=$(yq '.n_A // 2' "$CONFIG_FILE")
    n_B=$(yq '.n_B // 2' "$CONFIG_FILE")
    C=$(yq -o=json '.C // [[0, 1]]' "$CONFIG_FILE")
    n_noise_order=$(yq '.n_noise_order // 3' "$CONFIG_FILE")
    n_points=$(yq -o=json '.n_points // [11, 21, 11]' "$CONFIG_FILE")
    m=$(yq '.m // 100' "$CONFIG_FILE")
    q=$(yq '.q // 5' "$CONFIG_FILE")
    N=$(yq '.N // 50' "$CONFIG_FILE")
    db=$(yq '.dB // "sim.db"' "$CONFIG_FILE")
    debug=$(yq '.debug // false' "$CONFIG_FILE")
    epsilon=$(yq '.epsilon // 1e-10' "$CONFIG_FILE")
    LAMBDA=$(yq -o=json '.Lambda // [[1, 0], [0, 1]]' "$CONFIG_FILE")
    search=$(yq '.search // ""' "$CONFIG_FILE")
    bounds=$(yq -o=json '.bounds // ""' "$CONFIG_FILE")
fi

# --- Re-parse CLI overrides ---
set -- "$@"
while [ $# -gt 0 ]; do
    case "$1" in
        --n_states) n_states="$2"; shift 2 ;;
        --n_inputs) n_inputs="$2"; shift 2 ;;
        --n_outputs) n_outputs="$2"; shift 2 ;;
        --n_output) n_output="$2"; shift 2 ;;  # for compatibility
        --n_A) n_A="$2"; shift 2 ;;
        --n_B) n_B="$2"; shift 2 ;;
        --C) C="$2"; shift 2 ;;
        --n_noise_order) n_noise_order="$2"; shift 2 ;;
        --n_points) n_points="$2"; shift 2 ;;
        --m) m="$2"; shift 2 ;;
        --q) q="$2"; shift 2 ;;
        --N) N="$2"; shift 2 ;;
        --dB) db="$2"; shift 2 ;;
        --debug) debug=true; shift ;;
        --epsilon) epsilon="$2"; shift 2 ;;
        --Lambda) LAMBDA="$2"; shift 2 ;;
        --search) search="$2"; shift 2 ;;
        --bounds) bounds="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# --- Run Python module ---
python -m indirect_identification.sps \
    --n_states "$n_states" \
    --n_inputs "$n_inputs" \
    --n_outputs "$n_outputs" \
    --n_A "$n_A" \
    --n_B "$n_B" \
    --C "$C" \
    --n_noise_order "$n_noise_order" \
    --n_points "$n_points" \
    --m "$m" \
    --q "$q" \
    --N "$N" \
    --dB "$db" \
    $( [[ "$debug" == "true" ]] && echo "--debug" ) \
    --epsilon "$epsilon" \
    --Lambda "$LAMBDA" \
    --search "$search" \
    --bounds "$bounds"
