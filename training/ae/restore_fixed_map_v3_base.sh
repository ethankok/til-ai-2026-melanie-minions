#!/usr/bin/env bash
# Restore the fixed-map-v3 AE base checkpoint on Workbench and print the
# provenance-faithful fine-tune command.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

BACKUP_CKPT="${SOURCE_CHECKPOINT:-${HOME}/ae-checkpoints-backup/deployed-bc-v1.pt}"
DEPLOY_CKPT="ae/models/bc.pt"
TRAIN_CKPT="training/ae/checkpoints/fixed-map-v3-base.pt"
SEED="${AE_FINETUNE_SEED:-88}"
OUT_CKPT="${AE_FINETUNE_OUT:-training/ae/checkpoints/fixed-map-v3-finetune-v1.pt}"
IMAGE_TAG="${AE_FIXED_MAP_IMAGE:-melanie-minions-ae:ae-fixed-map-v3}"
REGISTRY_IMAGE="${AE_FIXED_MAP_REGISTRY_IMAGE:-asia-southeast1-docker.pkg.dev/til-ai-2026/repo-til-26-melanie-minions/melanie-minions-ae:ae-fixed-map-v3}"
N_FRAMES="${AE_FINETUNE_N_FRAMES:-}"

mkdir -p ae/models training/ae/checkpoints

restore_from_checkpoint() {
  local src="$1"
  if [[ ! -f "${src}" ]]; then
    return 1
  fi
  cp "${src}" "${DEPLOY_CKPT}"
  cp "${src}" "${TRAIN_CKPT}"
  echo "Restored fixed-map-v3 base checkpoint from ${src}"
  return 0
}

restore_from_image() {
  local image="$1"
  if ! command -v docker >/dev/null 2>&1; then
    return 1
  fi
  if ! docker image inspect "${image}" >/dev/null 2>&1; then
    return 1
  fi
  local cid
  cid="$(docker create "${image}")"
  trap 'docker rm -f "${cid}" >/dev/null 2>&1 || true' RETURN
  docker cp "${cid}:/workspace/models/bc.pt" "${TRAIN_CKPT}"
  cp "${TRAIN_CKPT}" "${DEPLOY_CKPT}"
  echo "Restored fixed-map-v3 base checkpoint from Docker image ${image}"
  return 0
}

restore_from_registry() {
  if ! command -v docker >/dev/null 2>&1; then
    return 1
  fi
  if ! docker pull "${REGISTRY_IMAGE}"; then
    return 1
  fi
  restore_from_image "${REGISTRY_IMAGE}"
}

if restore_from_checkpoint "${BACKUP_CKPT}"; then
  :
elif restore_from_image "${IMAGE_TAG}"; then
  :
elif restore_from_registry; then
  :
else
  cat >&2 <<EOF
Could not find the fixed-map-v3 checkpoint.

Tried:
  1. ${BACKUP_CKPT}
  2. ${IMAGE_TAG}:/workspace/models/bc.pt
  3. ${REGISTRY_IMAGE}:/workspace/models/bc.pt

Run this on the Workbench that originally built/submitted ae-fixed-map-v3, or set
SOURCE_CHECKPOINT=/path/to/deployed-bc-v1.pt.
EOF
  exit 1
fi

python - <<'PY'
from pathlib import Path
import hashlib

path = Path("training/ae/checkpoints/fixed-map-v3-base.pt")
print("checkpoint:", path)
print("sha256:", hashlib.sha256(path.read_bytes()).hexdigest())

try:
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print("n_frames:", ckpt.get("n_frames"))
    print("use_belief:", ckpt.get("use_belief"))
    print("epoch:", ckpt.get("epoch"))
    print("val_acc:", ckpt.get("val_acc"))
    print("ppo_eval_score:", ckpt.get("ppo_eval_score"))
except Exception as exc:
    print("torch metadata unavailable:", repr(exc))
PY

if [[ -z "${N_FRAMES}" ]]; then
  N_FRAMES="$(
    python - <<'PY'
from pathlib import Path

try:
    import torch
    ckpt = torch.load(
        Path("training/ae/checkpoints/fixed-map-v3-base.pt"),
        map_location="cpu",
        weights_only=False,
    )
    print(int(ckpt.get("n_frames", 1) or 1))
except Exception:
    print(1)
PY
  )"
fi

cat <<EOF

Fixed-map-v3 base is restored in both expected locations:
  ${DEPLOY_CKPT}
  ${TRAIN_CKPT}

Fine-tune from this exact base with seed ${SEED} and n_frames=${N_FRAMES}:

python training/ae/train_ppo.py \\
  --bc-checkpoint ${TRAIN_CKPT} \\
  --out ${OUT_CKPT} \\
  --updates 200 \\
  --games-per-update 12 \\
  --opponents scripted \\
  --eval-opponents scripted \\
  --novice \\
  --eval-every 5 \\
  --eval-games 30 \\
  --n-frames ${N_FRAMES} \\
  --seed ${SEED} \\
  --eval-seed ${SEED} \\
  2>&1 | tee fixed-map-v3-finetune-v1.log
EOF
