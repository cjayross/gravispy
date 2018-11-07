import gravispy.geom.geom as geom
__all__ = geom.__all__ + ['metric', 'lensing']
from gravispy.geom.geom import *
import gravispy.geom.metric as metric
import gravispy.geom.lensing as lensing

from numpy import seterr
seterr(all='print')
