#!/bin/env sh

python tools/load_image_data.py petimages cat dog --shape=50,50 $@
