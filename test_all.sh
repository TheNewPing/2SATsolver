#!/bin/bash
# test_all.sh: Run bin/main on all files in test_data/

set -e

BIN=bin/main
TESTDIR=test_data
N_SOL=2
MIN_DIST=1

if [ ! -x "$BIN" ]; then
    echo "Error: $BIN not found or not executable. Compile first."
    exit 1
fi

for f in $TESTDIR/*.txt; do
    echo "\n===== Running on $f ====="
    $BIN "$f" $N_SOL $MIN_DIST
    echo "========================\n"
done
