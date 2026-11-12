#!/usr/bin/env bash
# Install Ghidra into ~/opt/ghidra (no sudo). Idempotent: if a working
# install is already there we just print the env line and exit.
#
# Usage:
#   bash scripts/install_ghidra.sh [GHIDRA_VERSION]
# Default version: 11.3.1 (PUBLIC release). Requires JDK 21+ on PATH.
#
# After install, add this to .env:
#   GHIDRA_INSTALL_DIR=$HOME/opt/ghidra

set -euo pipefail

VERSION="${1:-11.3.1}"
RELEASE_TAG="Ghidra_${VERSION}_build"
# The actual zip is named with a build date, so we resolve via the GitHub
# redirect-to-asset endpoint.
DEST="$HOME/opt/ghidra"

# JDK check.
if ! command -v java >/dev/null 2>&1; then
  echo "ERROR: java not on PATH. Install JDK 21+ first (e.g. apt install openjdk-21-jdk)." >&2
  exit 1
fi
JVER=$(java -version 2>&1 | awk -F\" '/version/ {print $2}' | cut -d. -f1)
if [[ -z "$JVER" || "$JVER" -lt 21 ]]; then
  echo "ERROR: JDK 21+ required (have JDK $JVER). Install openjdk-21-jdk." >&2
  exit 1
fi

# Skip re-install.
if [[ -x "$DEST/support/analyzeHeadless" ]]; then
  echo "Ghidra already installed at $DEST"
  echo "GHIDRA_INSTALL_DIR=$DEST"
  exit 0
fi

mkdir -p "$HOME/opt"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "Resolving Ghidra ${VERSION} release URL..."
# GitHub redirects /releases/<tag>/download/Ghidra_<v>_PUBLIC_<datestamp>.zip
# but the datestamp isn't predictable. We use the API to find the asset.
ASSET_URL=$(curl -fsSL "https://api.github.com/repos/NationalSecurityAgency/ghidra/releases/tags/${RELEASE_TAG}" \
  | grep -E '"browser_download_url".*ghidra_'"${VERSION}"'_PUBLIC_.*\.zip"' \
  | head -n1 | awk -F'"' '{print $4}')
if [[ -z "$ASSET_URL" ]]; then
  echo "Could not find a release asset for ${RELEASE_TAG}." >&2
  echo "Browse https://github.com/NationalSecurityAgency/ghidra/releases and pass the right version." >&2
  exit 1
fi
echo "Downloading: $ASSET_URL"
curl -fSL -o "$TMP/ghidra.zip" "$ASSET_URL"
echo "Extracting..."
unzip -q "$TMP/ghidra.zip" -d "$TMP"

# The archive extracts a directory like ghidra_11.3.1_PUBLIC; symlink that as $DEST.
EXTRACTED=$(find "$TMP" -maxdepth 1 -type d -name "ghidra_${VERSION}_PUBLIC")
if [[ -z "$EXTRACTED" ]]; then
  echo "Unexpected archive layout under $TMP" >&2
  ls -la "$TMP" >&2
  exit 1
fi
mv "$EXTRACTED" "$DEST"

if [[ ! -x "$DEST/support/analyzeHeadless" ]]; then
  echo "Install looks broken: $DEST/support/analyzeHeadless missing" >&2
  exit 1
fi

echo
echo "Ghidra ${VERSION} installed to $DEST"
echo "Add this to your .env (or shell rc):"
echo "  GHIDRA_INSTALL_DIR=$DEST"
