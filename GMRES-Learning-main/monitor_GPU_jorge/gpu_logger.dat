#!/usr/bin/env bash
set -euo pipefail

INTERVAL="${1:-0.1}"         # segundos
OUTFILE="${2:-gpu_log.csv}"  # salida

echo "timestamp,idx,gpu_util_pct,mem_used_pct" > "$OUTFILE"

cleanup() { echo "# stopped $(date +%s.%N)" >> "$OUTFILE"; }
trap cleanup INT TERM

while true; do
  ts="$(date +%s.%N)"  # epoch en segundos con decimales (ideal para t_rel)
  nvidia-smi \
    --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits \
  | awk -v TS="$ts" -F',' '{
      for (i=1;i<=NF;i++) gsub(/^ +| +$/, "", $i);
      idx=$1; gpu=$2; used=$3; total=$4;
      mempct = (total>0) ? (100.0*used/total) : 0.0;
      # 2 decimales en memoria, GPU como entero (puedes cambiarlo si quieres)
      printf "%s,%s,%s,%.2f\n", TS, idx, gpu, mempct
    }' >> "$OUTFILE"

  sleep "$INTERVAL"
done

