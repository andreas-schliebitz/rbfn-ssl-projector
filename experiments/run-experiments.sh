#!/usr/bin/env bash

find ./runs -type f -name "*.sh" -print0 \
| sort -z \
| while IFS= read -r -d '' script; do
    echo "Running experiment: ${script}"
    time /bin/bash "${script}"
  done
