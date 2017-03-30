from __future__ import print_function
import ngmix

from ngmix.gexceptions import BootPSFFailure, BootGalFailure

class MiniMOF(dict):
    def __init__(self, config, allobs, prior=None):
        self.update(config)

        self.allobs=allobs
        self.nobj = len(allobs)
        self.prior=prior
    
    def get_result(self):
        """
        returns
        -------
        results: dict
            Dict with
            {'converged':bool,
             'results':[list of results]}
             'fitters':[list of fitters]}
        """
        return self._results

    def go(self):
        """
        parameters
        ----------
        allobs:
            Lists of ngmix Observation/ObsList/MultiBandObsList
        """

        while True:
            results = self._fit_all()
            if results['converged']:
                break

        self._results=results

    def _fit_one(self, i):
        """
        Fit one object, subtracting light from neighbors

        parameters
        ----------
        observation:
            ngmix Observation/ObsList/MultiBandObsList
        """

        # first time this might not have any correction
        obs = self._get_corrected_obs(i)

        boot = self._get_bootstrapper(obs)

        Tguess=4.0
        ppars=self['psf_pars']

        # will raise BootPSFFaiure
        boot.fit_psfs(ppars['model'], Tguess, ntry=ppars['ntry'])

        mconf=self['max_pars']
        covconf=mconf['cov']

        # will raise BootGalFailure
        boot.fit_max(self['fit_model'],
                     mconf,
                     prior=self.prior,
                     ntry=mconf['ntry'])


        fitter=boot.get_max_fitter() 
        return fitter



    def _get_bootstrapper(self, obs):
        """
        get the appropriate bootstrapper
        """

        return ngmix.bootstrap.Bootstrapper(obs)
