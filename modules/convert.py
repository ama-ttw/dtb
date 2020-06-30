from fractions import Fraction
import modules.settings as SETTINGS
import numpy as np
import cv2


def to_dtb_rot(num):
    num = Fraction(
        str(num))  # use str(num) to prevent floating point inaccuracies
    n, d = (num.numerator, num.denominator)
    m, p = divmod(abs(n), d)
    if n < 0:
        m = -m
    return '{}d{}f'.format(m, int(p/d*SETTINGS.FRAME)) if p > 0 \
        else '{}'.format(m)


def to_query_position(x, y, M):
    pt = cv2.perspectiveTransform(
        np.float32([x, y]).reshape(-1, 1, 2), M)
    pt = pt.flatten()
    return pt
