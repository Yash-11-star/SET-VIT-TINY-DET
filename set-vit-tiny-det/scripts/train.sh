#!/usr/bin/env bash
set -e
CFG=${1:-configs/coco_small.yaml}
python -m src.train --config $CFG
