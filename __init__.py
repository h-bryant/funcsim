import sys

# check python version
pyVersion = float(sys.version_info[0]) + 0.1 * float(sys.version_info[1])
if pyVersion < 3.5:
    raise Exception("Must be using Python >= 3.5")


import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from core import crossec
from core import recdyn

from rdarrays import fromcsv
from rdarrays import chron
from rdarrays import recall

from dependence import normal
from dependence import corru

import shrinkage


def shrink(a):
    # shrinkage cov estimator.
    # 'a' is a numpy array, with vars in cols & obs in rows
    return shrinkage.target_f(a)[0]