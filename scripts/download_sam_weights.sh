#!/usr/bin/env bash

set -euo pipefail

# Directory that contains this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_PATH="${REPO_ROOT}/change_detection/sam_vit_h_4b8939.pth"
TMP_PATH="${TARGET_PATH}.download"
SOURCE_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
EXPECTED_SHA256="a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e"

usage() {
  cat <<'EOF'
Download the SAM ViT-H checkpoint to change_detection/sam_vit_h_4b8939.pth.

USAGE:
  scripts/download_sam_weights.sh [--force]

OPTIONS:
  --force   Overwrite an existing sam_vit_h_4b8939.pth
EOF
}

FORCE_REDOWNLOAD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE_REDOWNLOAD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

download() {
  mkdir -p "$(dirname "${TARGET_PATH}")"
  echo "Downloading SAM weights from ${SOURCE_URL}"
  curl --fail --location --retry 3 --retry-delay 5 \
    --output "${TMP_PATH}" \
    "${SOURCE_URL}"
}

verify_sha() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${TMP_PATH}" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${TMP_PATH}" | awk '{print $1}'
  else
    echo "SKIP"  # No checksum tool available.
  fi
}

cleanup() {
  rm -f "${TMP_PATH}"
}
trap cleanup EXIT

if [[ -f "${TARGET_PATH}" ]]; then
  if [[ ${FORCE_REDOWNLOAD} -eq 0 ]]; then
    echo "SAM weights already exist at ${TARGET_PATH} (use --force to re-download)."
    exit 0
  fi
  rm -f "${TARGET_PATH}"
fi

download

CALC_SHA256="$(verify_sha)"
if [[ "${CALC_SHA256}" != "SKIP" ]]; then
  if [[ "${CALC_SHA256}" != "${EXPECTED_SHA256}" ]]; then
    echo "Checksum mismatch! Expected ${EXPECTED_SHA256} but found ${CALC_SHA256}" >&2
    exit 1
  fi
  echo "Checksum OK (${CALC_SHA256})"
else
  echo "Warning: Could not verify checksum because sha256sum/shasum is unavailable." >&2
fi

mv "${TMP_PATH}" "${TARGET_PATH}"
echo "Saved SAM weights to ${TARGET_PATH}"
