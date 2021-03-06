import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from core import static
from core import recdyn

from rdarrays import chron
from rdarrays import recall

from dependence import normal
from dependence import cgauss
from dependence import cstudent
from dependence import cgumbel
from dependence import cclayton
from dependence import fitkdemv

from shrinkage import shrink
from shrinkage import target_a
from shrinkage import target_b
from shrinkage import target_c
from shrinkage import target_d
from shrinkage import target_e
from shrinkage import target_f

from edf import makeedf

from kde import fitkde

from ecdfgof import kstest
from ecdfgof import adtest
from ecdfgof import cvmtest

from distfit import fit
from distfit import compare
