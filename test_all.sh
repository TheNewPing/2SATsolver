#!/bin/bash
# test_all.sh: Run bin/main on all files in test_data/

set -e

BIN=bin/main
TESTDIR=test_data

if [ ! -x "$BIN" ]; then
    echo "Error: $BIN not found or not executable. Compile first."
    exit 1
fi

passed_tests=()
failed_tests=()

for f in $TESTDIR/*.txt; do
    filename=$(basename "$f" .txt)
    # Extract N_SOL and MIN_DIST using pattern: test_n[n_sol]_d[min_dist]_[name]
    if [[ $filename =~ n([0-9]+)_d([0-9]+)_ ]]; then
        N_SOL="${BASH_REMATCH[1]}"
        MIN_DIST="${BASH_REMATCH[2]}"
    else
        echo "Warning: Could not extract N_SOL and MIN_DIST from $filename"
        continue
    fi
    echo "===== Running on $f (N_SOL=$N_SOL, MIN_DIST=$MIN_DIST) ====="
    output=$($BIN "$f" "$N_SOL" "$MIN_DIST")
    # echo "$output"
    if echo "$output" | grep -q "All solutions are valid."; then
        passed_tests+=("$filename")
        echo "Test $filename passed."
    else
        failed_tests+=("$filename")
        echo "Test $filename failed."
    fi
    # echo "========================"
done

echo
echo "Passed tests:"
for t in "${passed_tests[@]}"; do
    echo "  $t"
done

echo
echo "Failed tests:"
for t in "${failed_tests[@]}"; do
    echo "  $t"
done
