#!/bin/bash

docker build \
    --tag=tensorflow2.4 \
    --build-arg USER_ID="$(id -u)" \
    --build-arg GROUP_ID="$(id -g)" \
    --rm=true \
    --force-rm=true .