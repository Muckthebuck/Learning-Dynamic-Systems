#!/bin/bash

# Default parameters
SIM_TYPE="Cart-Pendulum"
T=10  # Simulation time in seconds
dT=0.02  # Time step
PLOT_ARG="--plot_system"  # Whether to plot the system
HISTORY_LIMIT=2.0  # Limit on history length
DB_FILE="sim_data.db"  # Database file
DIST_ARG=""
DISTURBANCE=50
# Usage function
usage() {
    echo "Usage: bash run_sims.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --sim <Pendulum|Cart-Pendulum|Carla>  Simulation type (default: Pendulum)"
    echo "  --T <float>                          Total simulation time (default: 10s)"
    echo "  --dt <float>                         Simulation time step (default: 0.02s)"
    echo "  --plot_system                        Enable/disable plotting (default: true)"
    echo "  --history_limit <float>                History limit for plotting (default: 2)"
    echo "  --dB <filename>                      Database file (default: sim_data.db)"
    echo "  --apply_disturbance                  Pass this flag to apply disturbance"
    echo "  --disturbance <float>                Disturbance Force (default: 50)"             
    echo "  -h, --help                           Show this help message"
    exit 0
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        --sim)
            SIM_TYPE="$2"
            shift 2
            ;;
        --T)
            T="$2"
            shift 2
            ;;
        --dt)
            dT="$2"
            shift 2
            ;;
        --plot_system)
            PLOT_ARG="--plot_system" 
            shift
            ;;
        --no-plot_system)
            PLOT_ARG="--no-plot_system" 
            shift
            ;;
        --apply_disturbance)
            DIST_ARG="--apply_disturbance" 
            shift
            ;;
        --disturbance)
            DISTURBANCE="$2"  
            shift
            ;;
        --history_limit)
            HISTORY_LIMIT="$2"
            shift 2
            ;;
        --dB)
            DB_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            usage
            ;;
    esac
done

# Activate virtual environment
source .venv/Scripts/activate

# Run the Python script
python -m sims.sims \
    --sim "$SIM_TYPE" \
    --T "$T" \
    --dt "$dT" \
    $PLOT_ARG \
    --history_limit "$HISTORY_LIMIT" \
    --dB "$DB_FILE" \
    $DIST_ARG \
    --disturbance "$DISTURBANCE"

# Deactivate virtual environment
deactivate
