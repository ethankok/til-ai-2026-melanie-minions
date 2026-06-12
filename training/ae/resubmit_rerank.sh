#!/usr/bin/env bash
# Eval is seeded/consistent -> ONE submit per image = the true score, so we
# submit one tag per DISTINCT model (skip the identical vf* duplicates).
#
# Run on the Workbench:  bash training/ae/resubmit_rerank.sh
# ~8 submits x 240s gap = ~28 min. Continues past any single failure.

set -u
TAGS=(
  # --- Tier 1: is the confpol NN even worth it vs the plain heuristic? ---
  tether-v1               # heuristic C+bomb7+tether, NO NN
  heuristic-c-bomb7-v1    # heuristic C+bomb7, no tether
  # --- Tier 2: can a confpol-native rung beat u860? ---
  confpol-native-u200     # confpol-native rung
  confpol-native-u360     # confpol-native rung
  confpol-native-u100     # early rung (inverted-U test)
  # --- Tier 3: field re-rank / sanity (opponents may have changed) ---
  heuristic-a-bomb7-v1    # heuristic-A profile
  pand-policy             # pure pandemonium policy
  pand-hybrid             # policy + heuristic safety veto
)

n=${#TAGS[@]}
for i in "${!TAGS[@]}"; do
  tag="${TAGS[$i]}"
  echo "============================================================"
  echo "[$((i+1))/$n] til submit ae ${tag}   ($(date +%H:%M:%S))"
  echo "============================================================"
  til submit ae "${tag}" || echo "!! submit failed for ${tag} — continuing"
  if [ "$i" -lt "$((n-1))" ]; then
    echo "--- sleeping 240s before next submit ---"
    sleep 240
  fi
done
echo "=== all ${n} re-rank submits dispatched ==="
