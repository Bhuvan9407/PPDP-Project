\
    #!/usr/bin/env bash
    set -euo pipefail

    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    cd "$ROOT"

    echo "=== KC-SLICE MILESTONE 2 SETUP ==="

    python -m pip install -q pandas numpy scikit-learn pyyaml

    # Mondrian upstream implementation
    mkdir -p methods/mondrian
    rm -rf methods/mondrian/upstream
    git clone --depth 1 https://github.com/qiyuangong/Mondrian.git \
        methods/mondrian/upstream

    # JVM
    if command -v apt-get >/dev/null 2>&1; then
      apt-get update -qq
      apt-get install -y -qq openjdk-21-jdk-headless
    fi

    # ARX library
    mkdir -p methods/arx
    rm -f methods/arx/libarx-3.9.2.jar
    wget -q \
      https://github.com/arx-deidentifier/arx/releases/download/v3.9.2/libarx-3.9.2.jar \
      -O methods/arx/libarx-3.9.2.jar

    # Adult
    mkdir -p data/raw data/smoke
    wget -q \
      https://huggingface.co/datasets/scikit-learn/adult-census-income/resolve/main/adult.csv \
      -O data/raw/adult.csv

    # Diabetes
    wget -q \
      "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip" \
      -O data/raw/diabetes.zip
    unzip -o data/raw/diabetes.zip -d data/raw >/dev/null

    # Initial KC-Slice structure
    mkdir -p methods/kc_slice
    touch methods/kc_slice/__init__.py
    touch methods/kc_slice/algorithm.py
    touch methods/kc_slice/config.py
    touch methods/kc_slice/io.py
    touch methods/kc_slice/evaluate.py
    touch methods/kc_slice/smoke_test.py

    echo "=== SETUP COMPLETE ==="
