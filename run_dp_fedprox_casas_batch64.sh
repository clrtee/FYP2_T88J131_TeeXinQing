#!/bin/bash
# run_dp_fedprox_casas_batch64.sh
# CASAS Aruba — batch=64, rounds=500, epochs=3, C=1.0, δ=1e-5
# n≈6880/client, q=0.0093, T=160500 steps
# All 10 clients run in parallel (requires ~10GB free RAM)

set -e
mkdir -p logs results

ALGO="fedprox"
ARCH="cnn"
PARTITION="non_iid"
ALPHA="0.1"
ROUNDS="500"
DATASET="casas"

run_exp() {
    local LABEL=$1
    local DP_FLAGS=$2

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  Dataset : CASAS  |  Privacy : $LABEL"
    echo "════════════════════════════════════════════════════"

    SERVER_LOG="logs/server_b64_casas_${LABEL}.log"
    CLIENT_LOG="logs/clients_b64_casas_${LABEL}.log"

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

    for i in $(seq 0 9); do
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
    done

    echo "  All 10 clients started."
    wait $SERVER_PID
    echo "  DONE: $LABEL (CASAS)"
    sleep 3
}

# ══════════════════════════════════════════════════════════════
# CASAS Aruba (n≈6880/client, q=0.0093, T=160500 steps)
# ε     σ
# 10    2.12
# 8     2.57
# 6     3.33
# 4     4.83
# 2     9.31
# ══════════════════════════════════════════════════════════════

echo ""
echo "████  CASAS Aruba  (batch=64, FedProx, 500 rounds)  ████"

run_exp "No_DP"         ""
run_exp "e10_VeryWeak"  "--dp --dp_noise 2.12 --dp_clip 1.0 --dp_epsilon 10.0"
run_exp "e8_Weak"       "--dp --dp_noise 2.57 --dp_clip 1.0 --dp_epsilon 8.0"
run_exp "e6_Medium"     "--dp --dp_noise 3.33 --dp_clip 1.0 --dp_epsilon 6.0"
run_exp "e4_Strong"     "--dp --dp_noise 4.83 --dp_clip 1.0 --dp_epsilon 4.0"
run_exp "e2_VeryStrong" "--dp --dp_noise 9.31 --dp_clip 1.0 --dp_epsilon 2.0"

echo ""
echo "CASAS all done!  Results: results/  Logs: logs/"
