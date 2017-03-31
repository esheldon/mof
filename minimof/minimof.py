from __future__ import print_function
try:
    xrange
except:
    xrange=range
    raw_input=input

import numpy
import ngmix

from ngmix.gexceptions import BootPSFFailure, BootGalFailure

class MiniMOF(dict):
    def __init__(self, config, allobs, Ttol=1.0e-4, maxiter=20, rng=None):
        self.update(config)

        self.Ttol=Ttol
        self.maxiter=maxiter

        if rng is None:
            rng=numpy.random.RandomState()
        self.rng=rng

        self.allobs=allobs
        self.nobj = len(allobs)
        self.prior=self._get_prior()
    
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

        self.first=True

        for iter in xrange(self.maxiter):
            print("iter: %d" % (iter+1))
            results = self._fit_all()

            self.first=False
            if results['converged']:
                break

        self._results=results

    def _fit_all(self):
        """
        run through and fit each observation
        """

        results={}
        reslist=[]
        for i in xrange(self.nobj):

            fitter=self._fit_one(i)
            res=fitter.get_result()
            reslist.append(res)

            ngmix.print_pars(res['pars'],front='  obj %d: ' % (i+1,))

        
        results['converged'] = self._check_convergence(reslist)

        return results

    def _check_convergence(self, reslist):
        if not hasattr(self, '_old_reslist'):
            self._old_reslist=reslist
            return False
        
        oreslist=self._old_reslist

        converged=True
        for ores,res in zip(oreslist, reslist):
            Told = ores['pars'][4]
            Tnew = res['pars'][4]

            Tdiff = abs(Tnew - Told)
            if Tdiff > self.Ttol:
                converged=False
                break

        self._old_reslist=reslist
        return converged

    def _fit_one(self, i):
        """
        Fit one object, subtracting light from neighbors

        parameters
        ----------
        observation:
            ngmix Observation/ObsList/MultiBandObsList
        """

        # first time this might not have any correction
        obs = self.get_corrected_obs(i)

        boot = self._get_bootstrapper(obs)

        Tguess=1.0
        ppars=self['psf_pars']

        # will raise BootPSFFaiure
        boot.fit_psfs(ppars['model'], Tguess, ntry=ppars['ntry'])

        mconf=self['max_pars']
        covconf=mconf['cov']

        # will raise BootGalFailure
        boot.fit_max(self['model'],
                     mconf,
                     prior=self.prior,
                     ntry=mconf['ntry'])

        fitter=boot.get_max_fitter() 

        gm = fitter.get_convolved_gmix()

        self.allobs[i][0].set_gmix(gm)

        return fitter


    def get_corrected_obs(self, i):
        """
        subtract all the neighbor light

        If this is the first time through, nothing is
        done
        """

        obslist = self.allobs[i]
        if self.first:
            return obslist

        obs=obslist[0]
        im = obs.image.copy()
        for iobs,tobslist in enumerate(self.allobs):
            if iobs==i:
                continue

            tobs = tobslist[0]
            model = tobs.gmix.make_image(
                im.shape,
                jacobian=tobs.jacobian,
            )
            
            im -= model

        if False:
            import images
            images.view_mosaic(
                [obs.image, im],
                label1='orig',
                label2='subtracted',
            )
            if 'q'==raw_input():
                stop

        nobs = ngmix.Observation(
            im,
            weight=obs.weight,
            jacobian=obs.jacobian,
            psf=obs.psf,
        )
        newlist=ngmix.ObsList(meta=obslist.meta)
        newlist.append(nobs)
        return newlist

    def _get_bootstrapper(self, obs):
        """
        get the appropriate bootstrapper
        """

        if self['model']=='cm':
            return ngmix.bootstrap.CompositeBootstrapper(
                obs,
                #fracdev_prior=self.fracdev_prior,
            )
        else:
            return ngmix.bootstrap.Bootstrapper(obs)

    def _get_prior(self):
        """
        Set all the priors
        """
        import ngmix
        from ngmix.joint_prior import PriorSimpleSep

        if 'priors' not in self:
            return None

        ppars=self['priors']

        gp = ppars['g']
        if gp['type']=='ba':
            g_prior = ngmix.priors.GPriorBA(gp['sigma'], rng=self.rng)
        elif gp['type']=='flat':
            g_prior = ngmix.priors.ZDisk2D(1.0, rng=self.rng)
        else:
            raise ValueError("implement other g prior")

        print("using input search prior")

        T_prior = None
        if 'T' in ppars:
            Tp = ppars['T']

            if Tp['type']=="flat":
                T_prior=ngmix.priors.FlatPrior(*Tp['pars'], rng=self.rng)

            elif Tp['type']=="gmixnd":
                T_prior = load_gmixnd(Tp, rng=self.rng)

            elif Tp['type']=='normal':
                Tpars=Tp['pars']
                T_prior=ngmix.priors.Normal(
                    Tp['mean'],
                    Tp['sigma'],
                    rng=self.rng,
                )

            elif Tp['type']=='lognormal':

                shift=Tp.get('shift',None)
                T_prior = ngmix.priors.LogNormal(
                    Tp['mean'],
                    Tp['sigma'],
                    shift=shift,
                    rng=self.rng,
                )

            elif Tp['type']=="two-sided-erf":
                T_prior_pars = Tp['pars']
                T_prior=ngmix.priors.TwoSidedErf(*T_prior_pars, rng=self.rng)
            else:
                raise ValueError("bad Tprior: '%s'" % Tp['type'])


        cp=ppars['counts']

        if cp['type']=="gmixnd":

            counts_prior = load_gmixnd(cp, rng=self.rng)

        elif cp['type']=='lognormal':

            counts_prior = ngmix.priors.LogNormal(
                cp['mean'],
                cp['sigma'],
                shift=cp.get('shift',None),
                rng=self.rng,
            )


        elif cp['type']=='normal':
            counts_prior=ngmix.priors.Normal(
                cp['mean'],
                cp['sigma'],
                rng=self.rng,
            )

        elif cp['type']=="two-sided-erf":
            counts_prior=ngmix.priors.TwoSidedErf(*cp['pars'], rng=self.rng)

        elif cp['type']=="flat":
            counts_prior=ngmix.priors.FlatPrior(*cp['pars'], rng=self.rng)

        else:
            raise ValueError("bad counts prior: '%s'" % cp['type'])

        cp=ppars['cen']
        if cp['type']=="truth":
            cen_prior=self.sim.cen_pdf
        elif cp['type'] == "normal2d":
            fit_cen_sigma=cp['sigma']
            cen_prior=ngmix.priors.CenPrior(
                0.0,
                0.0,
                fit_cen_sigma,
                fit_cen_sigma,
                rng=self.rng,
            )
        else:
            raise ValueError("bad cen prior: '%s'" % cp['type'])

        prior = PriorSimpleSep(cen_prior,
                               g_prior,
                               T_prior,
                               counts_prior)
        return prior


