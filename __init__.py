import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from core import static
from core import recdyn

from rdarrays import chron
from rdarrays import recall

from dependence import normal
from dependence import corru

from shrinkage import shrink

from helpers import fromcsv