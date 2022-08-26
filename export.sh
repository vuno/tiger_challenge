#!/usr/bin/env bash

./build.sh

docker save tigeralgorithm | gzip -c > tigeralgorithm.tar.xz
