#!/bin/bash
set -e
cd "$(dirname "$0")"

# whisper.cpp uses std::filesystem which requires macOS 10.15+
export CMAKE_OSX_DEPLOYMENT_TARGET=10.15
export MACOSX_DEPLOYMENT_TARGET=10.15

echo "=== Building Whisper Transcriber ==="
npm run tauri build -- --bundles app

APP="src-tauri/target/release/bundle/macos/Whisper Transcriber.app"
if [ -d "$APP" ]; then
  echo ""
  echo "Ad-hoc signing bundle..."
  codesign --force --deep --sign - "$APP"
  echo ""
  echo "Built: $APP"
  open "src-tauri/target/release/bundle/macos/"
else
  echo "Build did not produce .app bundle"
  exit 1
fi

echo ""
echo "Press any key to close..."
read -n 1
