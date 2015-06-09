#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

import numpy as np
import termcolor

import logging

def hinton_diagram(arr, max_arr=None):
    if max_arr is None:
        max_arr = arr
    max_val = max(abs(np.max(max_arr)), abs(np.min(max_arr)))
    return np.array2string(arr,
        formatter={'float_kind': lambda x: hinton_diagram_value(x,max_val)},
        max_line_width = 5000
    )

def hinton_diagram_value(val, max_val):
    chars = [ ' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█' ]
    if abs(abs(val) - max_val) < 1e-8:
        step = len(chars) - 1
    else:
        step = int(abs(float(val) / max_val) * len(chars))
    attr = 'dark' if val < 0 else 'bold'
    return termcolor.colored(chars[step], attrs=[attr])

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    a = np.random.randn(2, 10, 10)
    print(a)
    print(hinton_diagram(a))
