import os
import distutils
from distutils.core import setup

scripts=[
    'minimof-test',
]

scripts=[os.path.join('bin',s) for s in scripts]

setup(
    name="minimof", 
    packages=['minimof'],
    version="0.1.0",
    scripts=scripts,
)




