"""
quick test to see if the algorithm is working
"""
from __future__ import print_function
import numpy
import ngmix

from ngmix.gexceptions import BootPSFFailure, BootGalFailure

from . import minimof


class Tester(object):
    def __init__(self, nsim_conf_name, fit_config_name):
        import nsim

        nsim_conf = nsim.files.read_config(nsim_conf_name)
        nsim_conf['seed'] = numpy.random.randint(0,2**30)
        self.sim=nsim.sime.Sim(nsim_conf)

        self.fit_config=nsim.files.read_config(fit_config_name)

    def go(self, n=1):

        for i in xrange(n):
            print("-"*70)
            print("example:",i)
            allobs = self.sim()

            mm=minimof.MiniMOF(
                self.fit_config['mof'],
                allobs,
                rng=self.sim.rng,
            )
            mm.go()
            res = mm.get_result()
            if not res['converged']:
                print("did not converge")
            else:
                print("converged")


def test():
    t=Tester('sim-em01nbr','run-em01nbr-mcal-t01')
    res=t.go(100)


