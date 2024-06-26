import numpy as np

NZ_BINS = 4
NZ_PAIRS = int(NZ_BINS * (NZ_BINS + 1) / 2)
BIN_PAIRS = [(i, j) for i in range(NZ_BINS) for j in range(i, NZ_BINS)]
BP2INDEX = {bp: i for i, bp in enumerate(BIN_PAIRS)}
D2R = np.pi / 180.0
STR2DEG = 4 * np.pi * (180 / np.pi) ** 2
ARCMIN2RAD = 1.0 / 60.0 * np.pi / 180.0
FSKY = 416.0 / STR2DEG
