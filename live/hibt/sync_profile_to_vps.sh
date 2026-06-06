#!/usr/bin/env bash
# Sync local Chrome profile (cookies/login state) to VPS
# Run this AFTER you've done --login locally and logged in successfully
set -e

VPS="root@47.79.32.65"
REMOTE_DIR="/opt/hibt"
LOCAL_PROFILE="$(dirname "$0")/runtime/hibt-chrome-profile"

if [ ! -d "$LOCAL_PROFILE" ]; then
  echo "Error: local profile not found at $LOCAL_PROFILE"
  echo "Run '--login' locally first to create the profile."
  exit 1
fi

echo "=== Syncing Chrome profile to VPS ==="
echo "Source: $LOCAL_PROFILE"
echo "Target: $VPS:$REMOTE_DIR/live/runtime/hibt-chrome-profile/"

rsync -avz --delete \
  --exclude='CacheStorage/' \
  --exclude='Cache/' \
  --exclude='Code Cache/' \
  --exclude='GPUCache/' \
  --exclude='Service Worker/' \
  --exclude='*.log' \
  "$LOCAL_PROFILE/" "$VPS:$REMOTE_DIR/live/runtime/hibt-chrome-profile/"

echo ""
echo "Profile synced. The VPS Chrome will use your login session."
echo "Note: if session expires, re-run --login locally and sync again."
