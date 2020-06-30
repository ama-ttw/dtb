from fractions import Fraction
import modules.settings as SETTINGS


def to_dtb_rot(num):
    num = Fraction(
        str(num))  # use str(num) to prevent floating point inaccuracies
    n, d = (num.numerator, num.denominator)
    m, p = divmod(abs(n), d)
    if n < 0:
        m = -m
    return '{}d{}f'.format(m, int(p/d*SETTINGS.FRAME)) if p > 0 \
        else '{}'.format(m)
