#!/usr/bin/env bash
# Exit on error
set -o errexit
set -o nounset
set -o pipefail

STORAGE_DIR=/opt/render/project/.render
CHROME_DIR=$STORAGE_DIR/chrome

if [[ ! -d $CHROME_DIR ]]; then
  echo "...Downloading Chrome"
  mkdir -p $CHROME_DIR
  cd $CHROME_DIR

  if ! wget -P ./ https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb; then
    echo "Failed to download Google Chrome. Exiting."
    exit 1
  fi

  dpkg -x google-chrome-stable_current_amd64.deb $CHROME_DIR || {
    echo "Failed to extract Google Chrome package. Exiting."
    exit 1
  }

  rm google-chrome-stable_current_amd64.deb
  cd $HOME/project/src # Make sure we return to where we were
else
  echo "...Using Chrome from cache"
fi
