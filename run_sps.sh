#!/bin/bash

# Display usage function
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --n_states N       Number of states (default: 2)"
    echo "  --n_inputs N       Number of inputs (default: 1)"
    echo "  --n_output N       Number of outputs (default: 1)"
    echo "  --C C_MATRIX       C matrix as a list-like string (e.g., '[1, 2, 3; 4, 5, 6]')"
    echo "  --n_noise_order N  Noise order (default: 3)"
    echo "  --n_points N       Number of points as a list-like string (e.g., '[11, 21, 11]')"
    echo "  --m M              Number of samples (default: 100)"
    echo "  --q Q              Number of points (default: 5)"
    echo "  --N N              N samples (default: 50)"
    echo "  --db DB_FILE       Database file name (default: 'sim.db')"
    echo "  --debug            Enable debug mode (optional)"
    echo "  --epsilon EPSILON  Epsilon value for stability check (default: 1e-10)"
    echo "  --h                Show this help message"
    exit 1
}

# Default values for optional arguments
n_states=2
n_inputs=1
n_output=1
C="[0, 1]"
n_noise_order=3
n_points="[11, 21, 11]"
m=100
q=5
N=50
db="sim.db"
debug=false
epsilon=1e-10

# Parse the command-line arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --n_states)
            n_states="$2"
            shift 2
            ;;
        --n_inputs)
            n_inputs="$2"
            shift 2
            ;;
        --n_output)
            n_output="$2"
            shift 2
            ;;
        --C)
            C="$2"
            shift 2
            ;;
        --n_noise_order)
            n_noise_order="$2"
            shift 2
            ;;
        --n_points)
            n_points="$2"
            shift 2
            ;;
        --m)
            m="$2"
            shift 2
            ;;
        --q)
            q="$2"
            shift 2
            ;;
        --N)
            N="$2"
            shift 2
            ;;
        --db)
            db="$2"
            shift 2
            ;;
        --debug)
            debug=true
            shift
            ;;
        --epsilon)
            epsilon="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Run the Python script with the parsed arguments
python3 your_script_name.py \
    --n_states "$n_states" \
    --n_inputs "$n_inputs" \
    --n_output "$n_output" \
    --C "$C" \
    --n_noise_order "$n_noise_order" \
    --n_points "$n_points" \
    --m "$m" \
    --q "$q" \
    --N "$N" \
    --db "$db" \
    --debug "$debug" \
    --epsilon "$epsilon"
