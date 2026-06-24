#!/usr/bin/env bash

set -e

RAMDISK_SIZE="${1}"

if [[ -z "${VIRTUAL_ENV}" ]]; then
    >&2 echo "No active virtual environment found."
    exit 1
fi

pip install --upgrade pip
pip install --upgrade --force-reinstall -e .
pip install pytest==9.0.1 black==25.11.0


if [[ -n "${RAMDISK_SIZE}" ]]; then
	mkdir -p /tmp/datasets
	mount -t tmpfs -o size="${RAMDISK_SIZE}" tmpfs /tmp/datasets
	mount | grep datasets
else
	echo "Warning: No RAMDISK_SIZE provided. Ramdisk for storing datasets is not created."
fi
