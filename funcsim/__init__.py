import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from core import simulate

from dependence import covtocorr
from dependence import spearman
from dependence import MvNorm
from dependence import MvKde
from dependence import CopulaGauss
from dependence import CopulaStudent
from dependence import CopulaClayton
from dependence import CopulaGumbel
from dependence import CopulaFrank

from imanconover import imanconover

from shrinkage import shrink
from nearby import nearestpd

from edf import edf

from kde import Kde

from ecdfgof import kstest
from ecdfgof import adtest
from ecdfgof import cvmtest

from shapiro import swtest

from distfit import fit
from distfit import compare

from utests import utests

from screen import screen

from cpt import cpt, utilPower, utilNormLog, weightTK, weightPrelec1
from cpt import weightPrelec2, cptBV

from eut import eut, utilIsoelastic, eutIsoelastic

from plotting import fan, twofuncs, histpdf, dblscat, qqplot, show

def version():
    """
    Return the current version of the funcsim package.

    Returns
    -------
    str
        The version string of the package (e.g., '0.1.5').
    """
    return "0.1.16"
