"""

for lots of crowding, the thing that dominates the chi squared
is the detections, not the model that's being fit or the
tolerance used for convergence

Using coarse tolerance helps a lot for getting the big groups,
and doesn't degrade the chi squared
{'ftol':tol, 'xtol':tol}

"""
import numpy as np
import ngmix
from .priors import PriorSimpleSepMulti

class Sim(dict):
    def __init__(self, config, seed):
        self.rng=np.random.RandomState(seed)

        gseed=self.rng.randint(0,2**30)
        self.guess_rng=np.random.RandomState(gseed)

        self.update(config)
        self['dims'] = np.array(self['dims'])
        self['psf_dims']=np.array( self['psf_dims'] )

        self.g_pdf = self._make_g_pdf()
        self.T_pdf = self._make_T_pdf()
        self.F_pdf = self._make_F_pdf()

        cen=(self['dims']-1.0)/2.0
        maxrad=cen[0]-self['dims'][0]/10.0
        print("maxrad:",maxrad)

        sigma=maxrad/3.0
        self.position_pdf=ngmix.priors.TruncatedSimpleGauss2D(
            cen[0], cen[1],
            sigma, sigma,
            maxrad,
        )

    def _make_g_pdf(self, prior=False):
        c=self['pdfs']['g']
        if prior:
            rng=self.guess_rng
        else:
            rng=self.rng

        return ngmix.priors.GPriorBA(c['sigma'], rng=rng)

    def _make_T_pdf(self, prior=False):
        c=self['pdfs']['T']
        return self._get_generic_pdf(c, prior=prior)

    def _make_F_pdf(self, prior=False):
        c=self['pdfs']['F']
        if c['type']=='trackT':
            if prior:
                cprior={}
                cprior.update(self['pdfs']['T'])
                assert cprior['type']in ['lognormal','normal']

                cprior['mean'] = cprior['mean']*c['factor']
                cprior['sigma'] = cprior['sigma']*c['factor']

                return self._get_generic_pdf(cprior, prior=True)
            else:
                return 'trackT'
        else:
            return self._get_generic_pdf(c, prior=prior)


    def _get_generic_pdf(self, c, prior=False):
        if prior:
            rng=self.guess_rng
        else:
            rng=self.rng


        if c['type']=='lognormal':
            pdf = ngmix.priors.LogNormal(
                c['mean'],
                c['sigma'],
                rng=rng,
            )
        elif c['type']=='flat':
            pdf = ngmix.priors.FlatPrior(
                c['range'][0],
                c['range'][1],
                rng=rng,
            )
        else:
            raise ValueError("bad pdf: '%s'" % c['type'])

        limits=c.get('limits',None)
        if prior or limits is None:
            return pdf
        else:
            return ngmix.priors.LimitPDF(pdf, [0.0, 30.0])

    def make_obs(self):
        self._set_psf()

        self._set_gmixes()

        self._make_image()
        self._draw_objects()
        self._add_noise()

        self._make_obs()

    def get_obs(self):
        return self.obs

    def get_image(self):
        return self.image

    def show(self):
        images.multiview(self.image)

    def get_npars_per(self):
        return len(self.parlist[0])

    def get_prior(self, objects=None, use_truth=False):
        """
        prior for N objects.  The priors are the same for
        structural parameters, the only difference being the
        centers
        """
        cen_priors=[]

        cen_sigma=1.0 #pixels for now
        if objects is not None:
            for i in range(len(objects)):
                p=ngmix.priors.CenPrior(
                    objects['y'][i], objects['x'][i],
                    cen_sigma, cen_sigma,
                    rng=self.guess_rng,
                )
                cen_priors.append(p)

        else:
            for i,pars in enumerate(self.parlist):
                cen1,cen2 = pars[0:2]
                p=ngmix.priors.CenPrior(
                    cen1, cen2,
                    cen_sigma, cen_sigma,
                    rng=self.rng,
                )
                cen_priors.append(p)

        if use_truth:
            g_prior = self._make_g_pdf(prior=True)
            T_prior = self._make_T_pdf(prior=True)
            F_prior = self._make_F_pdf(prior=True)
        else:
            g_prior=ngmix.priors.GPriorBA(0.2,rng=self.guess_rng)
            T_prior = ngmix.priors.TwoSidedErf(-1.0, 0.1, 1000.0, 1.0,
                                               rng=self.guess_rng)
            #T_prior = ngmix.priors.FlatPrior(-0.5, 1000.0, rng=self.rng)
            F_prior = ngmix.priors.TwoSidedErf(-1.0, 0.1, 1000.0, 1.0,
                                               rng=self.guess_rng)

        return PriorSimpleSepMulti(
            cen_priors,
            g_prior,
            T_prior,
            F_prior,
        )


    def get_guess(self, objects=None, from_truth=False, from_pdf=False):
        """
        we need some kind of decent start for the
        position and the flux, so we always base these
        on the truth.  But the flux is allowed to range
        pretty wide

        from_truth means we also make a close guess at
        the size and flux based on the truth
        """
        npars_per = self.get_npars_per()
        rng=self.rng

        if objects is not None:
            #if from_truth or from_pdf:
            if from_truth:
                raise ValueError("don't set from_truth or from_pdf when "
                                 "sending objects")
            num=len(objects)
        else:
            num=len(self.parlist)

        npars_tot = num*npars_per
        guess = np.zeros(npars_tot)

        assert npars_per==6,"6 for now"
        assert npars_tot==6*num

        for i in range(num):
            beg=i*npars_per

            if objects is not None:
                row, col = objects['y'][i], objects['x'][i]
                F = objects['flux'][i]
                T = objects['x2'][i] + objects['y2'][i]
            else:
                pars=self.parlist[i]
                row, col = pars[0:2]
                T = pars[4]
                F = pars[5]

            # always close guess for center
            guess[beg+0] = row + rng.uniform(low=-0.05, high=0.05)
            guess[beg+1] = col + rng.uniform(low=-0.05, high=0.05)

            # always arbitrary guess for shape
            guess[beg+2] = rng.uniform(low=-0.05, high=0.05)
            guess[beg+3] = rng.uniform(low=-0.05, high=0.05)

            if from_truth:
                guess[beg+4] = T*( 1.0 + rng.uniform(low=-0.1, high=0.1) )
                guess[beg+5] = F*( 1.0 + rng.uniform(low=-0.1, high=0.1) )

            elif from_pdf:
                #g1,g2 = self.g_pdf.sample2d()
                #guess[beg+2] = g1
                #guess[beg+3] = g2

                # guess T and F from the pdfs
                T = self.T_pdf.sample()
                #T = self.T_pdf.mean*(1.0 + rng.uniform(low=-0.1, high=0.1))
                if self.F_pdf=='trackT':
                    F = self.T_pdf.sample()
                    F *= self['pdfs']['F']['factor']
                else:
                    F = self.F_pdf.sample()

                guess[beg+4] = T
                guess[beg+5] = F

            else:

                # arbitrary guess for size
                #guess[beg+4] = 2.0*(1.0 + rng.uniform(low=-0.1, high=0.1))
                #r=rng.uniform(
                #    low=-0.1*self.T_pdf.sigma,
                #    high=0.1*self.T_pdf.sigma,
                #)
                #guess[beg+4] = self.T_pdf.mean*(1.0 + r)

                guess[beg+4] = T*(1.0 + rng.uniform(low=-0.05, high=0.05))
                guess[beg+5] = F*(1.0 + rng.uniform(low=-0.05, high=0.05))

        return guess

    def _set_psf(self):
        pars = [0.0,0.0,0.0,0.0,4.0,1.0]
        self.psf = ngmix.GMixModel(pars,"gauss")

        pcen=(self['psf_dims']-1.0)/2.0
        pjac = ngmix.UnitJacobian(row=pcen[0], col=pcen[1])

        self.psf_im = self.psf.make_image(self['psf_dims'], jacobian=pjac)

        self.psf_im += self.rng.normal(
            scale=self['psf_noise_sigma'],
            size=self['psf_dims'],
        )
        psf_wt=np.zeros(self['psf_dims'])+1.0/self['psf_noise_sigma']**2

        self.psf_obs = ngmix.Observation(
            self.psf_im,
            weight=psf_wt,
            jacobian=pjac,
            gmix=self.psf,
        )

    def _make_image(self):
        self.image0 = np.zeros( self['dims'], dtype='f8')

        # this is for drawing
        self.jacobian = ngmix.UnitJacobian(row=0, col=0)

    def _get_gmix(self, pars):
        """
        draw a random model
        """

        i=self.rng.randint(0, len(self['models']))
        modname=self['models'][i]

        if modname=='bulge+disk':
            fracdev = self.rng.uniform(low=0.0, high=1.0)
            TdByTe=1.0
            gm0 = ngmix.gmix.GMixCM(fracdev, TdByTe, pars)
        else:
            gm0 = ngmix.GMixModel(pars, modname)

        gm = gm0.convolve(self.psf)
        return gm

    def _set_gmixes(self):
        rng=self.rng
        obj_pars = []

        #cen=(self['dims']-1.0)/2.0
        #low=cen-self['dims']/4.0
        #high=cen+self['dims']/4.0
        gmlist=[]
        parlist=[]

        for i in range(self['nobj']):

            #obj_cen1 = rng.uniform(low=low[0], high=high[0])
            #obj_cen2 = rng.uniform(low=low[1], high=high[1])
            obj_cen1, obj_cen2 = self.position_pdf.sample()

            g1,g2 = self.g_pdf.sample2d()
            T = self.T_pdf.sample()
            if self.F_pdf=='trackT':
                F = T*self['pdfs']['F']['factor']
            else:
                F = self.F_pdf.sample()

            pars = [obj_cen1, obj_cen2, g1, g2, T, F]
            print("TF:",T,F)

            gm=self._get_gmix(pars)

            gmlist.append(gm)
            parlist.append(pars)

        self.gmlist=gmlist
        self.parlist=parlist

    def _draw_objects(self):
        """
        this is dumb, drawing into the full image when
        we don't need to
        """

        for gm in self.gmlist:
            tim = gm.make_image(self['dims'], jacobian=self.jacobian)

            self.image0 += tim

    def _add_noise(self):
        self.image = self.image0 + \
                self.rng.normal(scale=self['noise_sigma'], size=self['dims'])


    def _make_obs(self):

        wt=np.zeros(self['dims']) + 1.0/self['noise_sigma']**2
        self.obs = ngmix.Observation(
            self.image,
            weight=wt,
            jacobian=self.jacobian,
            psf=self.psf_obs,
        )


