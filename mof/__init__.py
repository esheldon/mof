# flake8: noqa

__version__ = '0.9.11'

from . import moflib
from .moflib import MOF, MOFStamps, MOFFlux
from .galsimfit import GSMOF, KGSMOF

from . import priors
from . import procflags

# test of big version
from . import moftest

from . import stamps
from . import fofs
from . import tests
