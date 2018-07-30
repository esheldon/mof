import os
import distutils
from distutils.core import setup

try:
    # for python 3, let 2to3 do most of the work
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    # for python 2 don't apply any transformations
    from distutils.command.build_py import build_py


scripts=[
    'minimof-test',
]

scripts=[os.path.join('bin',s) for s in scripts]

setup(
    name="minimof", 
    packages=['minimof'],
    version="0.1.0",
    scripts=scripts,
    cmdclass={'build_py': build_py},
)




