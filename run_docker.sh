#!/bin/bash

docker run --rm -it \
--gpus all \
-p 8888:8888 \
-v "$(pwd):/workdir" \
-w /workdir \
tensorflow2.4 \
bash