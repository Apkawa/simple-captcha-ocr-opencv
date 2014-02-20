# -*- coding: utf-8 -*-
import os

from pyfann import libfann

from test_opencv import PROJECT_ROOT


imgW = 83
imgH = 23

factor = 2

rimgW = imgW * factor
rimgH = imgH * factor

nSegs = 5

segW = 18
segH = 18

segSize = (segW, segH)

num_input = segW * segH

TRAIN_FILE = os.path.join(PROJECT_ROOT, "fann.train")


