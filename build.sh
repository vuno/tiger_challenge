#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build \
    --build-arg UID="$(id -u ${USER})" \
    -t tigeralgorithm "$SCRIPTPATH"
