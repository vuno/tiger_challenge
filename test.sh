#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

SEGMENTATION_FILE="/output/images/breast-cancer-segmentation-for-tils/segmentation.tif"
DETECTION_FILE="/output/detected-lymphocytes.json"
TILS_SCORE_FILE="/output/til-score.json"

MEMORY=32g

echo "Building docker"
./build.sh

echo "Creating volume..."
docker volume create tiger-output

echo "Running algorithm..."
docker run --rm \
        --memory=$MEMORY \
        --memory-swap=$MEMORY \
        --network=none \
        --cap-drop=ALL \
        --security-opt="no-new-privileges" \
        --shm-size=128m \
        --pids-limit=256 \
        --gpus=all \
        -v $SCRIPTPATH/testinput/:/input/ \
        -v $SCRIPTPATH/testoutput/:/output/ \
        tigeralgorithm

echo "Checking output files..."
docker run --rm \
        -v $SCRIPTPATH/testoutput:/output/ \
        python:3.8-slim \
        python -m json.tool $DETECTION_FILE; \
        /bin/bash; \
        [[ -f $SEGMENTATION_FILE ]] || printf 'Expected file %s does not exist!\n' "$SEGMENTATION_FILE"; \
        [[ -f $TILS_SCORE_FILE ]] || printf 'Expected file %s does not exist!\n' "$TILS_SCORE_FILE"; \

echo "Removing volume..."
docker volume rm tiger-output
