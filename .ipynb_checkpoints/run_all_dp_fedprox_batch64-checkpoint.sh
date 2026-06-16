#!/bin/bash
# run_all_dp_fedprox_batch64.sh
# batch=64, rounds=500, epochs=3, C=1.0
# Runs 2 clients at a time to avoid OOM (15GB RAM system)

set -e
mkdir -p logs results

ALGO="fedprox"
ARCH="cnn"
PARTITION="non_iid"
ALPHA="0.1"
ROUNDS="500"

run_exp() {
    local DATASET=$1
    local LABEL=$2
    local DP_FLAGS=$3

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  Dataset : ${DATASET^^}  |  Privacy : $LABEL"
    echo "════════════════════════════════════════════════════"

    SERVER_LOG="logs/server_b64_${DATASET}_${LABEL}.log"
    CLIENT_LOG="logs/clients_b64_${DATASET}_${LABEL}.log"

    # Start server
    python -m src.main \
        --role server \
        --dataset "$DATASET" \
        --arch "$ARCH" \
        --algo "$ALGO" \
        --partition "$PARTITION" \
        --alpha "$ALPHA" \
        --rounds "$ROUNDS" \
        $DP_FLAGS \
        > "$SERVER_LOG" 2>&1 &

    SERVER_PID=$!
    echo "  Server PID=$SERVER_PID  log=$SERVER_LOG"
    sleep 5

    # 2 clients at a time to avoid OOM
    for i in 0 1 2 3 4 5 6 7 8 9; do
        python -m src.main \
            --role client \
            --cid "$i" \
            --dataset "$DATASET" \
            --arch "$ARCH" \
            --algo "$ALGO" \
            --partition "$PARTITION" \
            --alpha "$ALPHA" \
            $DP_FLAGS \
            >> "$CLIENT_LOG" 2>&1 &

        # Every 2 clients, wait for them to finish before starting next 2
        if (( (i+1) % 2 == 0 )); then
            wait
            echo "  Clients $((i-1)) and $i done, continuing..."
            sleep 1
        fi
    done

    wait $SERVER_PID
    echo "  DONE: $LABEL ($DATASET)"
    sleep 3
}

# ══ UCI HAR (n≈735, q=0.0871, T=16500) ══
echo ""; echo "████  UCI HAR  (batch=64)  ████"

run_exp "uci" "No_DP"         ""
run_exp "uci" "e10_VeryWeak"  "--dp --dp_noise 6.35  --dp_clip 1.0 --dp_epsilon 10.0"
run_exp "uci" "e8_Weak"       "--dp --dp_noise 7.72  --dp_clip 1.0 --dp_epsilon 8.0"
run_exp "uci" "e6_Medium"     "--dp --dp_noise 9.99  --dp_clip 1.0 --dp_epsilon 6.0"
run_exp "uci" "e4_Strong"     "--dp --dp_noise 14.49 --dp_clip 1.0 --dp_epsilon 4.0"
run_exp "uci" "e2_VeryStrong" "--dp --dp_noise 27.95 --dp_clip 1.0 --dp_epsilon 2.0"

# ══ CASAS (n≈6880, q=0.0093, T=160500) ══
echo ""; echo "████  CASAS Aruba  (batch=64)  ████"

run_exp "casas" "No_DP"         ""
run_exp "casas" "e10_VeryWeak"  "--dp --dp_noise 2.12 --dp_clip 1.0 --dp_epsilon 10.0"
run_exp "casas" "e8_Weak"       "--dp --dp_noise 2.57 --dp_clip 1.0 --dp_epsilon 8.0"
run_exp "casas" "e6_Medium"     "--dp --dp_noise 3.33 --dp_clip 1.0 --dp_epsilon 6.0"
run_exp "casas" "e4_Strong"     "--dp --dp_noise 4.83 --dp_clip 1.0 --dp_epsilon 4.0"
run_exp "casas" "e2_VeryStrong" "--dp --dp_noise 9.31 --dp_clip 1.0 --dp_epsilon 2.0"

echo ""; echo "All done! Results: results/  Logs: logs/"