from __future__ import print_function
try:
    xrange
except:
    xrange=range
    raw_input=input

import numpy
import ngmix
from ngmix.bootstrap import Bootstrapper, CompositeBootstrapper

from ngmix.gexceptions import BootPSFFailure, BootGalFailure

from ngmix.observation import Observation, ObsList, MultiBandObsList

class MiniMOF(dict):
    def __init__(self, config, allobs, rng=None):
        self.update(config)

        if rng is None:
            rng=numpy.random.RandomState()
        self.rng=rng

        self.set_allobs(allobs)
        self.nobj = len(self.allobs)
        self.prior=self._get_prior()
    
    def set_allobs(self, allobs):
        """
        currently assume internally we got a list of  ObsLists

        WE will convert list of Observations here.  Need to 
        support MultiBandObsList too
        """
        if isinstance(allobs[0], ObsList):
            self.allobs=allobs
        elif isinstance(allobs[0], Observation):
            nlist = []
            for obs in allobs:
                obslist = ObsList()
                obslist.append( obs )
                nlist.append(obslist)

            self.allobs = nlist
        else:
            raise ValueError("currently only lists of "
                             "Observation or Obslist supported")

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

        for iter in xrange(self['maxiter']):
            print("iter: %d" % (iter+1))

            if iter > 1:
                fix_center=False
            else:
                fix_center=True

            results = self._fit_all(fix_center=fix_center)

            self.first=False
            if results['converged']:
                break

        self._results=results

    def _fit_all(self, fix_center=False):
        """
        run through and fit each observation
        """

        results={}
        reslist=[]
        for i in xrange(self.nobj):

            fitter=self._fit_one(i, fix_center=fix_center)
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

            Tdiff = abs(Tnew - Told)/Told
            print("fracdiff:",Tdiff)
            if Tdiff > self['Ttol']:
                converged=False
                break

        self._old_reslist=reslist
        return converged

    def _fit_one(self, i, fix_center=False):
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

        if fix_center:
            self.prior.cen_prior_save=self.prior.cen_prior
            self.prior.cen_prior=ngmix.priors.CenPrior(
                0.0,0.0,
                0.0001,0.0001,
            )

        # will raise BootGalFailure
        boot.fit_max(
            self['model'],
            mconf,
            prior=self.prior,
            ntry=mconf['ntry'],
        )
        if fix_center:
            self.prior.cen_prior=self.prior.cen_prior_save

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
                titles=['orig', 'subtracted'],
                title='object %d' % (i+1),
            )
            if 'q'==raw_input('hit a key: '):
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

    def get_model_obs(self):
        """
        full model image
        """

        obslist = self.allobs[0]
        obs = obslist[0]
        im=0*obs.image
        for iobs,tobslist in enumerate(self.allobs):

            tobs = tobslist[0]
            model = tobs.gmix.make_image(
                im.shape,
                jacobian=tobs.jacobian,
            )
            
            im += model

        nobs = ngmix.Observation(
            im,
            weight=obs.weight,
            jacobian=obs.jacobian,
            psf=obs.psf,
        )
        newlist=ngmix.ObsList(meta=obslist.meta)
        newlist.append(nobs)
        return newlist

    def show_residuals(self, **kw):
        """
        show residuals
        """
        import images
        im = self.allobs[0][0].image
        model_obs = self.get_model_obs()

        plt=images.compare_images(
            im,
            model_obs[0].image,
            label1='original',
            label2='model',
            **kw
        )
        return plt

    def _get_bootstrapper(self, obs):
        """
        get the appropriate bootstrapper
        """

        if self['model']=='cm':
            return CompositeBootstrapper(
                obs,
                fracdev_prior=self.fracdev_prior,
            )
        else:
            return Bootstrapper(obs)

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

            elif Tp['type'] in ['two-sided-erf','TwoSidedErf']:
                T_prior = ngmix.priors.TwoSidedErf(
                    *Tp['pars'],
                    rng=self.rng
                )


            elif Tp['type']=='lognormal':

                shift=Tp.get('shift',None)
                T_prior = ngmix.priors.LogNormal(
                    Tp['mean'],
                    Tp['sigma'],
                    shift=shift,
                    rng=self.rng,
                )

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

        elif cp['type']=='TwoSidedErf':
            counts_prior = ngmix.priors.TwoSidedErf(
                *cp['pars'],
                rng=self.rng
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


        if 'fracdev' in ppars:
            fp = ppars['fracdev']
            means = numpy.array(fp['means'])
            weights = numpy.array(fp['weights'])
            covars= numpy.array(fp['covars'])

            if len(means.shape) == 1:
                means = means.reshape( (means.size, 1) )
            if len(covars.shape) == 1:
                covars = covars.reshape( (covars.size, 1, 1) )

            self.fracdev_prior = ngmix.gmix.GMixND(
                weights,
                means,
                covars,
            )

        return prior


