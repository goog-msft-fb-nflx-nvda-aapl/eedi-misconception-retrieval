#!/usr/bin/env bash
# needs the kaggle cli + token, and the eedi competition rules accepted (late submission is fine)
set -e
mkdir -p data && cd data
kaggle competitions download -c eedi-mining-misconceptions-in-mathematics
unzip -o eedi-mining-misconceptions-in-mathematics.zip
