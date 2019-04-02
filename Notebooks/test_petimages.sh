#!/bin/env sh

python ui/image_load.py --model=keras_petimages_cnn.model --shape=50,50,3 --labels=petimages.l.pickle $@
