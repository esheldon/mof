"""
todo

    - PSF fit and stats (use k space for that)
    - s/n
    - understand why getting too big fft errors.  Maybe need to
      limit hlr? Or is it due to no longer limiting fracdev?

    - guesses are still an issue I think. More testing with injections.
"""
from __future__ import print_function
import numpy as np
import ngmix
from ngmix.gexceptions import GMixRangeError
from ngmix.fitting import run_leastsq
from ngmix.priors import LOWVAL

from .moflib import MOFStamps, DEFAULT_LM_PARS

FOLDING_THRESHOLD = 0.05

class KGSMOF(MOFStamps):
    """
    version using galsim for modelling, and doing convolutions by multiplying
    in fourier space
    """
    def __init__(self, list_of_obs, model, prior, **keys):
        """
        list_of_obs is not an ObsList, it is a python list of
        Observation/ObsList/MultiBandObsList
        """
        import galsim

        self._gsp = galsim.GSParams(folding_threshold=FOLDING_THRESHOLD)

        self._set_all_obs(list_of_obs)
        self._setup_nbrs()
        self.model = model

        self._set_model_maker()

        self.prior = prior

        self.nobj = len(self.list_of_obs)

        if model == 'bdf':
            self.npars_per = 6+self.nband
            self.nband_pars_per = 7

            # center1 + center2 + shape + hlr + fracdev + fluxes
            # for each object
            self.n_prior_pars = self.nobj*(1 + 1 + 1 + 1 + 1 + self.nband)
        else:
            self.npars_per = 5+self.nband
            self.nband_pars_per = 6
            self._band_pars = np.zeros(self.nband_pars_per*self.nobj)

            # center1 + center2 + shape + hlr + fluxes for each object
            self.n_prior_pars = self.nobj*(1 + 1 + 1 + 1 + self.nband)

        self.npars = self.nobj*self.npars_per

        self.lm_pars = {}
        self.lm_pars.update(DEFAULT_LM_PARS)

        lm_pars = keys.get('lm_pars', None)
        if lm_pars is not None:
            self.lm_pars.update(lm_pars)

        if 'maxfev' not in self.lm_pars:
            # default in leastsq is 100*(self.npars+1)
            self.lm_pars['maxfev'] = 300*(self.npars+1)

        self._init_model_images()
        self._set_fdiff_size()

    def go(self, guess):
        """
        Run leastsq and set the result
        """

        guess = np.array(guess, dtype='f8', copy=False)

        nobj = guess.size//self.npars_per
        nleft = guess.size % self.npars_per
        if nobj != self.nobj or nleft != 0:
            raise ValueError("bad guess size: %d" % guess.size)

        bounds = self._get_bounds(nobj)

        result = run_leastsq(
            self._calc_fdiff,
            guess,
            self.n_prior_pars,
            k_space=True,
            bounds=bounds,
            **self.lm_pars
        )

        result['model'] = self.model
        if result['flags'] == 0:
            stat_dict = self.get_fit_stats(result['pars'])
            result.update(stat_dict)

        self._result = result

    def _calc_fdiff(self, pars):
        """
        vector with (model-data)/error.

        The npars elements contain -ln(prior)
        """
        import galsim

        # we cannot keep sending existing array into leastsq, don't know why
        fdiff = np.zeros(self.fdiff_size)
        start = 0

        try:

            for iobj, mbo in enumerate(self.list_of_obs):
                # fill priors and get new start
                objpars = self.get_object_pars(pars, iobj)
                start = self._fill_priors(objpars, fdiff, start)

                for band, obslist in enumerate(mbo):
                    band_pars = self.get_object_band_pars(
                        pars,
                        iobj,
                        band,
                    )

                    for obs in obslist:

                        central_model = self._make_fully_shifted_model(
                            band_pars,
                            obs,
                        )

                        meta = obs.meta
                        kimage = meta['kimage']
                        kmodel = meta['kmodel']
                        ierr = meta['ierr']
                        psf_ii = meta['psf_ii']

                        maxrad = self._get_maxrad(obs)
                        nbr_models = self._get_nbr_models(
                            iobj,
                            pars,
                            meta,
                            band,
                            maxrad,
                            obs,
                        )

                        if len(nbr_models) > 0:
                            all_models = [central_model] + nbr_models

                            total_model = galsim.Add(
                                all_models,
                                gsparams=self._gsp,
                            )
                        else:
                            total_model = central_model

                        total_model = galsim.Convolve(
                            total_model,
                            psf_ii,
                            gsparams=self._gsp,
                        )
                        total_model.drawKImage(image=kmodel)

                        kmodel -= kimage

                        # (model-data)/err
                        kmodel.array.real[:, :] *= ierr
                        kmodel.array.imag[:, :] *= ierr

                        # now copy into the full fdiff array
                        imsize = kmodel.array.real.size

                        fdiff[start:start+imsize] = kmodel.array.real.ravel()

                        start += imsize

                        fdiff[start:start+imsize] = kmodel.array.imag.ravel()

                        start += imsize

        except GMixRangeError:
            fdiff[:] = LOWVAL

        return fdiff

    def _make_fully_shifted_model(self, band_pars, obs):
        """
        get the model with the relative shift, but also the shift
        for jacobian center
        """
        central_model0 = self.make_model(band_pars)
        dv, du = self._get_jcen_shift(obs)
        central_model = central_model0.shift(du, dv)
        return central_model

    def _get_maxrad(self, obs):
        """
        criteria for now is that the object is within
        a stamp radii.  Not great at all

        This is done because off chip objects seem to cause
        a big problem for the k space fitter
        """
        scale = obs.meta['scale']
        return obs.image.shape[0]*scale*0.5

    def _get_nbr_models(self, iobj, pars, meta, band, maxrad, obs):
        models = []
        for nbr in meta['nbr_data']:
            # assert nbr['index'] != iobj
            nbr_pars = self.get_object_band_pars(
                pars,
                nbr['index'],
                band,
            )

            # the current pars [v,u,..] are relative to
            # fiducial position.  we need to add these to
            # the fiducial for the rendering within
            # the stamp of the central
            rad_offset = np.sqrt(
                nbr['v0']**2 + nbr['u0']**2
            )
            if rad_offset < maxrad:
                nbr_pars[0] += nbr['v0']
                nbr_pars[1] += nbr['u0']
                nbr_model = self._make_fully_shifted_model(nbr_pars, obs)
                models.append(nbr_model)

        return models

    def _get_jcen_shift(self, obs):
        """
        for galsim need an extra shift for the fact that the jacobian
        center may be off from the canonical center
        """
        jac = obs.jacobian
        cen = jac.cen

        # now see how far we need to shift in v,u to get the
        # given row offset from the canonical center

        ccen = (np.array(obs.image.shape)-1.0)/2.0
        cdiff = cen - ccen
        dv, du = jac.get_vu(cen[0] + cdiff[0], cen[1] + cdiff[1])
        return dv, du

    def _fill_priors(self, pars, fdiff, start):
        """
        same prior for every object
        """

        nprior = self.prior.fill_fdiff(pars, fdiff[start:])

        return start+nprior

    def make_model(self, pars):
        """
        make the galsim model
        """

        model = self.make_round_model(pars)

        dy, dx = pars[0:0+2]
        g1 = pars[2]
        g2 = pars[3]

        g = np.sqrt(g1**2 + g2**2)
        if g > 0.99:
            raise GMixRangeError('g too big')

        # argh another generic error
        try:
            model = model.shear(g1=g1, g2=g2)
        except (RuntimeError, ValueError) as err:
            raise GMixRangeError(str(err))

        model = model.shift(dx, dy)
        return model

    def make_round_model(self, pars):
        """
        make the round galsim model, unshifted
        """

        kw = {}
        hlr = pars[4]

        if self.model in ['bdf', 'dev']:
            # if hlr > self.max_dim_arcsec/2:
            if hlr > self.max_dim_arcsec/4:
                kw['trunc'] = self.max_dim_arcsec*2
                if self.model == 'dev':
                    kw['flux_untruncated'] = True

        if self.model == 'bdf':
            kw['fracdev'] = pars[5]
            flux = pars[6]
        else:
            flux = pars[5]

        # this throws a generic runtime error so there is no way to tell what
        # went wrong

        try:
            model = self._model_maker(
                half_light_radius=hlr,
                flux=flux,
                gsparams=self._gsp,
                **kw
            )
        except RuntimeError as err:
            raise GMixRangeError(str(err))

        return model

    def _set_model_maker(self):
        import galsim
        if self.model == 'exp':
            self._model_maker = galsim.Exponential
        elif self.model == 'dev':
            self._model_maker = galsim.DeVaucouleurs
        elif self.model == 'gauss':
            self._model_maker = galsim.Gaussian
        elif self.model == 'bdf':
            self._model_maker = make_bdf
        else:
            raise NotImplementedError("can't fit '%s'" % self.model)

    def _set_all_obs(self, list_of_obs):
        for i, mbobs in enumerate(list_of_obs):
            nb = len(mbobs)
            if i == 0:
                self.nband = nb
            else:
                assert nb == self.nband,\
                    'all obs must have same number of bands'

        self.list_of_obs = list_of_obs

    def _init_model_images(self):
        """
        model images for each observation will be added to meta
        """
        import galsim

        max_dim = 0
        totpix = 0
        for mbobs in self.list_of_obs:
            for obslist in mbobs:
                for obs in obslist:
                    jac = obs.jacobian
                    pjac = obs.psf.jacobian

                    gsimage = galsim.Image(
                        obs.image,
                        wcs=jac.get_galsim_wcs()
                    )
                    psf_gsimage = galsim.Image(
                        obs.psf.image,
                        wcs=pjac.get_galsim_wcs()
                    )

                    ii = galsim.InterpolatedImage(gsimage)
                    psf_ii = galsim.InterpolatedImage(psf_gsimage)

                    meta = obs.meta
                    meta['scale'] = jac.scale
                    meta['gsimage'] = gsimage
                    meta['psf_ii'] = psf_ii
                    meta['kimage'] = ii.drawKImage()
                    meta['kmodel'] = meta['kimage'].copy()

                    weight = meta['kimage'].real.array.copy()
                    weight[:, :] = 0.5*obs.weight.max()

                    # parseval's theorem
                    weight *= (1.0/weight.size)

                    ierr = weight.copy()
                    ierr[:, :] = 0.0

                    w = np.where(weight > 0)
                    if w[0].size > 0:
                        ierr[w] = np.sqrt(weight[w])

                    meta['ierr'] = ierr

                    dim = max(obs.image.shape)*jac.scale
                    max_dim = max(max_dim, dim)

                    totpix += meta['kimage'].array.size

        self.max_dim_arcsec = max_dim
        self.totpix = totpix

    def _set_fdiff_size(self):
        """
        we have 2*totpix, since we use both real and imaginary
        parts
        """

        self.fdiff_size = self.n_prior_pars + 2*self.totpix

    def get_object_s2n(self, i):
        """
        TODO: implement for galsim

        get the s/n for the given object.  This uses just the model
        to calculate the s/n, but does use the full weight map
        """

        s2n_sum = 0.0
        mbobs = self.list_of_obs[i]
        for band, obslist in enumerate(mbobs):
            for obsnum, obs in enumerate(obslist):
                im = self.make_image(i, band=band, obsnum=obsnum, include_nbrs=False)
                wt = obs.weight

                s2n_sum += (im**2 * wt).sum()

        if s2n_sum < 0.0:
            s2n_sum = 0.0

        return np.sqrt(s2n_sum)

    def get_object_psf_stats(self, i):
        """
        TODO: implement for galsim psfs?  how can we get the psfs
        we fit from ngmix?
        """
        return {
            'g': [0.0, 0.0],
            'T': -9999.0,
        }
        raise NotImplementedError('not yet implemented')

        g1sum = 0.0
        g2sum = 0.0
        Tsum = 0.0
        wsum = 0.0

        mbobs = self.list_of_obs[i]
        for band, obslist in enumerate(mbobs):
            for obsnum, obs in enumerate(obslist):
                twsum = obs.weight.sum()
                wsum += twsum

                tg1, tg2, tT = obs.psf.gmix.get_g1g2T()

                g1sum += tg1*twsum
                g2sum += tg2*twsum
                Tsum += tT*twsum

        g1 = g1sum/wsum
        g2 = g2sum/wsum
        T = Tsum/wsum

        return {
            'g': [g1, g2],
            'T': T,
        }

    def get_object_result(self, i):
        """
        get a result dict for a single object
        """
        pars = self._result['pars']
        pars_cov = self._result['pars_cov']

        pres = self.get_object_psf_stats(i)

        res = {}

        res['nband'] = self.nband
        res['psf_g'] = pres['g']
        res['psf_T'] = pres['T']

        res['nfev'] = self._result['nfev']
        res['s2n'] = self.get_object_s2n(i)
        res['pars'] = self.get_object_pars(pars, i)
        res['pars_cov'] = self.get_object_cov(pars_cov, i)
        res['g'] = res['pars'][2:2+2].copy()
        res['g_cov'] = res['pars_cov'][2:2+2, 2:2+2].copy()
        res['hlr'] = res['pars'][4]
        res['hlr_err'] = np.sqrt(res['pars_cov'][4, 4])

        if self.model == 'bdf':
            res['fracdev'] = res['pars'][5]
            res['fracdev_err'] = np.sqrt(res['pars_cov'][5, 5])
            flux_start = 6
        else:
            flux_start = 5

        res['flux'] = res['pars'][flux_start:]
        res['flux_cov'] = res['pars_cov'][flux_start:, flux_start:]
        res['flux_err'] = np.sqrt(np.diag(res['flux_cov']))

        return res

    def make_corrected_image(self, index, band=0, obsnum=0):
        """
        TODO: implement for galsim

        get an observation for the given object and band
        with all the neighbors subtracted from the image
        """
        raise NotImplementedError('not yet implemented')

        pars = self.get_result()['pars']

        ref_obs = self.list_of_obs[index][band][obsnum]
        psf_gmix = ref_obs.psf.gmix
        jacob = ref_obs.jacobian

        image = ref_obs.image.copy()

        nbr_data = ref_obs.meta['nbr_data']
        if len(nbr_data) > 0:
            for nbr in nbr_data:
                nbr_pars = self.get_object_band_pars(
                    pars,
                    nbr['index'],
                    band,
                )

                # the current pars [v,u,..] are relative to
                # fiducial position.  we need to add these to
                # the fiducial for the rendering within
                # the stamp of the central

                nbr_pars[0] += nbr['v0']
                nbr_pars[1] += nbr['u0']

                gm0 = self._make_model(nbr_pars)
                gm = gm0.convolve(psf_gmix)

                modelim = gm.make_image(image.shape, jacobian=jacob)

                image -= modelim

        return image

    def get_fit_stats(self, pars):
        return {}

    def make_image(self, iobj, band=0, obsnum=0, include_nbrs=False):
        """
        make an image for the given band and observation number

        including nbrs is causing some trouble, probably due to
        the object being way off the stamp (why does it work for
        the fitter?)
        """
        import galsim

        res = self.get_result()
        pars = res['pars']

        obs = self.list_of_obs[iobj][band][obsnum]
        meta = obs.meta

        band_pars = self.get_object_band_pars(
            pars,
            iobj,
            band,
        )

        central_model = self._make_fully_shifted_model(band_pars, obs)

        if include_nbrs:
            maxrad = self._get_maxrad(obs)
            nbr_models = self._get_nbr_models(
                iobj,
                pars,
                meta,
                band,
                maxrad,
                obs
            )
        else:
            nbr_models = []

        if len(nbr_models) > 0:
            all_models = [central_model] + nbr_models

            total_model = galsim.Add(
                all_models,
                gsparams=self._gsp,
            )
        else:
            total_model = central_model

        convolved_model = galsim.Convolve(
            total_model,
            meta['psf_ii'],
            gsparams=self._gsp,
        )
        image = meta['gsimage'].copy()
        convolved_model.drawImage(
            image=image,
            method='no_pixel',
        )
        return image.array


class GSMOF(KGSMOF):
    """
    slow real space fitter
    """
    def _calc_fdiff(self, pars):
        """
        vector with (model-data)/error.

        The npars elements contain -ln(prior)
        """
        import galsim
        from galsim import GalSimFFTSizeError

        # we cannot keep sending existing array into leastsq, don't know why
        fdiff = np.zeros(self.fdiff_size)
        start = 0

        try:

            for iobj, mbo in enumerate(self.list_of_obs):
                # fill priors and get new start
                objpars = self.get_object_pars(pars, iobj)
                start = self._fill_priors(objpars, fdiff, start)

                for band, obslist in enumerate(mbo):
                    band_pars = self.get_object_band_pars(
                        pars,
                        iobj,
                        band,
                    )

                    for obs in obslist:

                        meta = obs.meta
                        model = meta['model']
                        ierr = meta['ierr']
                        wpos = meta['wpositive']

                        central_model = self.make_model(band_pars)

                        maxrad = self._get_maxrad(obs)
                        nbr_models = self._get_nbr_models(
                            iobj,
                            pars,
                            meta,
                            band,
                            maxrad,
                            obs,
                        )

                        if len(nbr_models) > 0:
                            all_models = [central_model] + nbr_models

                            total_model = galsim.Add(
                                all_models,
                                gsparams=self._gsp,
                            )
                        else:
                            total_model = central_model

                        total_model = galsim.Convolve(
                            total_model,
                            obs.psf.meta['ii'],
                            gsparams=self._gsp,
                        )

                        self._do_draw(obs, total_model, model)

                        # (model-data)/err
                        tfdiff = model.array
                        tfdiff -= obs.image
                        tfdiff *= ierr

                        # now copy into the full fdiff array
                        # imsize = tfdiff.size
                        wsize = wpos[0].size

                        fdiff[start:start+wsize] = tfdiff[wpos].ravel()

                        start += wsize

        except (GMixRangeError, GalSimFFTSizeError):
            fdiff[:] = LOWVAL

        return fdiff

    def _do_draw(self, obs, obj, gsimage):
        """
        draw into the image with an offset
        """
        jac = obs.jacobian

        # note reverse for galsim
        ccen = (np.array(obs.image.shape) - 1.0)/2.0

        jrow, jcol = jac.get_cen()
        offset = (jcol - ccen[1], jrow - ccen[0])

        obj.drawImage(
            image=gsimage,
            method='no_pixel',
            offset=offset,
        )

    def get_object_s2n(self, i):
        """
        TODO: implement for galsim

        get the s/n for the given object.  This uses just the model
        to calculate the s/n, but does use the full weight map
        """

        s2n_sum = 0.0
        mbobs = self.list_of_obs[i]
        for band, obslist in enumerate(mbobs):
            for obsnum, obs in enumerate(obslist):
                im = self.make_image(i, band=band, obsnum=obsnum, include_nbrs=False)
                wt = obs.weight

                s2n_sum += (im**2 * wt).sum()

        if s2n_sum < 0.0:
            s2n_sum = 0.0

        return np.sqrt(s2n_sum)

    def _get_maxrad(self, obs):
        """
        no problem with off-stamp neighbors for real space fitting
        """
        return 1.e9

    def _set_all_obs(self, list_of_obs):
        self.list_of_obs = list_of_obs
        for i, mbo in enumerate(list_of_obs):
            if i == 0:
                self.nband = len(mbo)
            else:
                assert len(mbo) == self.nband, \
                    'all obs must have same number of bands'

    def _set_fdiff_size(self):
        self.fdiff_size = self.n_prior_pars + self.totpix

    def _init_model_images(self):
        """
        model images for each observation will be added to meta
        """

        totpix = 0
        max_dim = 0
        for mbobs in self.list_of_obs:
            for obslist in mbobs:
                for obs in obslist:
                    meta = obs.meta
                    jac = obs.jacobian

                    weight = obs.weight
                    ierr = weight.copy()
                    ierr[:, :] = 0.0

                    w = np.where(weight > 0)
                    if w[0].size > 0:
                        ierr[w] = np.sqrt(weight[w])

                    meta['ierr'] = ierr
                    meta['scale'] = jac.scale
                    meta['wpositive'] = w
                    self._create_models_in_obs(obs)

                    dim = max(obs.image.shape)*jac.scale
                    max_dim = max(max_dim, dim)

                    # totpix += weight.size
                    totpix += w[0].size

        self.max_dim_arcsec = max_dim
        self.totpix = totpix

    def _create_models_in_obs(self, obs):
        import galsim

        psf_gsimage = galsim.Image(
            obs.psf.image/obs.psf.image.sum(),
            wcs=obs.psf.jacobian.get_galsim_wcs(),
        )
        psf_ii = galsim.InterpolatedImage(
            psf_gsimage,
        )

        gsimage = galsim.Image(
            obs.image.copy(),
            wcs=obs.jacobian.get_galsim_wcs(),
        )

        meta = obs.meta
        meta['model'] = gsimage
        obs.psf.meta['ii'] = psf_ii

    def make_image(self, iobj, band=0, obsnum=0, include_nbrs=False):
        """
        make an image for the given band and observation number

        including nbrs is causing some trouble, probably due to
        the object being way off the stamp (why does it work for
        the fitter?)
        """
        import galsim

        res = self.get_result()
        pars = res['pars']

        obs = self.list_of_obs[iobj][band][obsnum]
        meta = obs.meta
        psf_meta = obs.psf.meta

        band_pars = self.get_object_band_pars(
            pars,
            iobj,
            band,
        )

        central_model = self._make_fully_shifted_model(band_pars, obs)

        if include_nbrs:
            maxrad = self._get_maxrad(obs)
            nbr_models = self._get_nbr_models(
                iobj,
                pars,
                meta,
                band,
                maxrad,
                obs,
            )
        else:
            nbr_models = []

        if len(nbr_models) > 0:
            all_models = [central_model] + nbr_models

            total_model = galsim.Add(
                all_models,
                gsparams=self._gsp,
            )
        else:
            total_model = central_model


        total_model = galsim.Convolve(
            total_model,
            psf_meta['ii'],
            gsparams=self._gsp,
        )

        image = meta['model'].copy()
        self._do_draw(obs, total_model, image)
        return image.array


class GSMOFFlux(GSMOF):
    """
    flux only fitter
    """
    def __init__(self, list_of_obs, model, **keys):
        import galsim

        self._gsp = galsim.GSParams(folding_threshold=FOLDING_THRESHOLD)

        self._set_all_obs(list_of_obs)
        self._setup_nbrs()
        self.model = model

        self._set_model_maker()

        self.nobj = len(self.list_of_obs)

        self.npars_per = self.nband
        self.nband_pars_per = 1

        # fluxes for each object
        self.n_prior_pars = 0

        self.npars = self.nobj*self.npars_per

        if model == 'bdf':
            self.nband_pars_per_full = 7
        else:
            self.nband_pars_per_full = 6

        self.lm_pars = {}
        self.lm_pars.update(DEFAULT_LM_PARS)

        lm_pars = keys.get('lm_pars', None)
        if lm_pars is not None:
            self.lm_pars.update(lm_pars)

        if 'maxfev' not in self.lm_pars:
            # default in leastsq is 100*(self.npars+1)
            self.lm_pars['maxfev'] = 300*(self.npars+1)

        self._init_model_images()
        self._set_fdiff_size()

    def _calc_fdiff(self, pars):
        """
        vector with (model-data)/error.

        The npars elements contain -ln(prior)
        """
        import galsim
        from galsim import GalSimFFTSizeError

        # we cannot keep sending existing array into leastsq, don't know why
        fdiff = np.zeros(self.fdiff_size)
        start = 0

        try:

            for iobj, mbo in enumerate(self.list_of_obs):
                # fill priors and get new start
                objpars = self.get_object_pars(pars, iobj)
                start = self._fill_priors(objpars, fdiff, start)

                for band, obslist in enumerate(mbo):
                    if 'summed_image' not in obslist[0].meta:
                        band_pars = self.get_object_band_pars(
                            pars,
                            iobj,
                            band,
                        )

                    else:

                        flux = self.get_object_band_flux(
                            pars,
                            iobj,
                            band,
                        )

                    for obs in obslist:

                        meta = obs.meta
                        model = meta['model']
                        ierr = meta['ierr']

                        if 'summed_image' not in meta:

                            central_model = self._make_fully_shifted_model(
                                band_pars,
                                obs,
                            )

                            central_image = model.copy()
                            convolved_model = galsim.Convolve(
                                central_model,
                                obs.psf.meta['ii'],
                                gsparams=self._gsp,
                            )
                            convolved_model.drawImage(
                                image=central_image,
                                method='no_pixel',
                            )

                            maxrad = self._get_maxrad(obs)
                            nbr_models = self._get_nbr_models(
                                iobj,
                                pars,
                                meta,
                                band,
                                maxrad,
                                obs,
                            )

                            central_image = central_image.array
                            summed_image = central_image.copy()

                            nbr_images = []
                            for nbr_model in nbr_models:
                                nbr_image = model.copy()

                                convolved_model = galsim.Convolve(
                                    nbr_model,
                                    obs.psf.meta['ii'],
                                    gsparams=self._gsp,
                                )
                                convolved_model.drawImage(
                                    image=nbr_image,
                                    method='no_pixel',
                                )

                                nbr_image = nbr_image.array
                                nbr_images.append(nbr_image)

                                summed_image += nbr_image

                            meta['central_image'] = central_image
                            meta['nbr_images'] = nbr_images
                            meta['summed_image'] = summed_image
                        else:

                            central_image = meta['central_image']
                            summed_image = meta['summed_image']

                            central_image *= flux/central_image.sum()
                            summed_image[:, :] = central_image

                            maxrad = self._get_maxrad(obs)
                            if len(meta['nbr_images']) > 0:
                                nbr_fluxes = \
                                    self._get_nbr_fluxes(
                                        iobj,
                                        pars,
                                        meta,
                                        band,
                                        maxrad,
                                    )
                                nim = meta['nbr_images']
                                for inbr, nbr_image in enumerate(nim):
                                    nbr_flux = nbr_fluxes[inbr]

                                    nbr_image *= nbr_flux/nbr_image.sum()

                                    summed_image[:, :] += nbr_image

                        # (model-data)/err
                        tfdiff = summed_image
                        tfdiff -= obs.image
                        tfdiff *= ierr

                        # now copy into the full fdiff array
                        imsize = tfdiff.size

                        fdiff[start:start+imsize] = tfdiff.ravel()

                        start += imsize

        except (GMixRangeError, GalSimFFTSizeError):
            fdiff[:] = LOWVAL

        return fdiff

    def get_object_band_flux(self, flux_pars, iobj, band):
        """
        get the input pars plus the flux
        """

        ind = iobj*self.npars_per + band

        flux = flux_pars[ind]

        if flux < 0.0:
            raise GMixRangeError('flux less than zero')

        return flux

    def get_object_band_pars(self, flux_pars, iobj, band):
        """
        get the input pars plus the flux
        """

        flux = self.get_object_band_flux(flux_pars, iobj, band)

        flags = self.list_of_obs[iobj].meta['input_flags']
        if flags == 0:
            pars = self.list_of_obs[iobj].meta['input_model_pars']
            pars = pars[0:self.nband_pars_per_full].copy()
        else:
            print('    filling a star for missing pars')
            pars = np.zeros(self.nband_pars_per_full)
            pars[4] = 1.0e-5

        pars[-1] = flux
        return pars

    def _get_nbr_fluxes(self, iobj, pars, meta, band, maxrad):
        fluxes = []
        for nbr in meta['nbr_data']:
            flux = self.get_object_band_flux(
                pars,
                nbr['index'],
                band,
            )
            fluxes.append(flux)

        return fluxes

    def _fill_priors(self, pars, fdiff, start):
        """
        no priors for this fitter
        """
        return start

    def get_object_result(self, i):
        """
        get a result dict for a single object
        """
        pars = self._result['pars']
        pars_cov = self._result['pars_cov']

        pres = self.get_object_psf_stats(i)

        res = {}

        res['nband'] = self.nband
        res['psf_g'] = pres['g']
        res['psf_T'] = pres['T']

        res['nfev'] = self._result['nfev']
        res['s2n'] = self.get_object_s2n(i)
        res['pars'] = self.get_object_pars(pars, i)
        res['pars_cov'] = self.get_object_cov(pars_cov, i)

        res['flux'] = res['pars'].copy()
        res['flux_cov'] = res['pars_cov'].copy()
        res['flux_err'] = np.sqrt(np.diag(res['flux_cov']))

        return res


class GSMOFFluxReRender(GSMOF):
    """
    flux only fitter
    """
    def __init__(self, list_of_obs, model, **keys):

        self._gsp = galsim.GSParams(folding_threshold=FOLDING_THRESHOLD)

        self._set_all_obs(list_of_obs)
        self._setup_nbrs()
        self.model = model

        self._set_model_maker()

        self.nobj = len(self.list_of_obs)

        self.npars_per = self.nband
        self.nband_pars_per = 1

        # fluxes for each object
        self.n_prior_pars = 0

        self.npars = self.nobj*self.npars_per

        if model == 'bdf':
            self.nband_pars_per_full = 7
        else:
            self.nband_pars_per_full = 6

        self.lm_pars = {}
        self.lm_pars.update(DEFAULT_LM_PARS)

        lm_pars = keys.get('lm_pars', None)
        if lm_pars is not None:
            self.lm_pars.update(lm_pars)

        if 'maxfev' not in self.lm_pars:
            # default in leastsq is 100*(self.npars+1)
            self.lm_pars['maxfev'] = 300*(self.npars+1)

        self._init_model_images()
        self._set_fdiff_size()

    def get_object_band_pars(self, flux_pars, iobj, band):
        """
        get the input pars plus the flux
        """

        ind = iobj*self.npars_per + band

        flux = flux_pars[ind]

        if flux < 0.0:
            raise GMixRangeError('flux less than zero')

        flags = self.list_of_obs[iobj].meta['input_flags']
        if flags == 0:
            pars = self.list_of_obs[iobj].meta['input_model_pars']
            pars = pars[0:self.nband_pars_per_full].copy()
        else:
            print('    filling a star for missing pars')
            pars = np.zeros(self.nband_pars_per_full)
            pars[4] = 1.0e-5

        pars[-1] = flux
        return pars

    def _fill_priors(self, pars, fdiff, start):
        """
        no priors for this fitter
        """
        return start

    def get_object_result(self, i):
        """
        get a result dict for a single object
        """
        pars = self._result['pars']
        pars_cov = self._result['pars_cov']

        pres = self.get_object_psf_stats(i)

        res = {}

        res['nband'] = self.nband
        res['psf_g'] = pres['g']
        res['psf_T'] = pres['T']

        res['nfev'] = self._result['nfev']
        res['s2n'] = self.get_object_s2n(i)
        res['pars'] = self.get_object_pars(pars, i)
        res['pars_cov'] = self.get_object_cov(pars_cov, i)

        res['flux'] = res['pars'].copy()
        res['flux_cov'] = res['pars_cov'].copy()
        res['flux_err'] = np.sqrt(np.diag(res['flux_cov']))

        return res


def make_bdf(half_light_radius=None,
             flux=None,
             fracdev=None,
             trunc=0.0,
             gsparams=None):

    """
    a bulge+disk maker for equal hlr for bulge and disk

    Parameters
    ----------
    half_light_radius: float
        hlr
    flux: float
        Total flux in the profile
    fracdev: float
        fraction of light in the bulge
    """
    import galsim

    assert half_light_radius is not None, 'send half_light_ratio'
    assert flux is not None, 'send flux'
    assert fracdev is not None, 'send fracdev'

    if trunc > 0:
        flux_untruncated = True
    else:
        flux_untruncated = False

    """
    bulge = galsim.DeVaucouleurs(
        half_light_radius=half_light_radius,
        flux=fracdev,
        flux_untruncated=flux_untruncated ,
        trunc=trunc,
        gsparams=gsparams,
    )
    """
    bulge = galsim.Sersic(
        n=2,
        half_light_radius=half_light_radius,
        flux=fracdev,
        flux_untruncated=flux_untruncated ,
        trunc=trunc,
        gsparams=gsparams,
    )

    disk = galsim.Exponential(
        half_light_radius=half_light_radius,
        flux=(1-fracdev),
        gsparams=gsparams,
    )

    return galsim.Add(
        bulge,
        disk,
        gsparams=gsparams,
    ).withFlux(flux)


# these are just examples, users should write their own
def get_mof_stamps_prior_gs(list_of_obs, model, rng):
    """
    Not generic, need to let this be configurable
    """

    nband = len(list_of_obs[0])

    obs = list_of_obs[0][0][0]
    cen_sigma = obs.jacobian.get_scale()  # a pixel
    cen_prior = ngmix.priors.CenPrior(
        0.0,
        0.0,
        cen_sigma, cen_sigma,
        rng=rng,
    )

    g_prior = ngmix.priors.GPriorBA(
        0.2,
        rng=rng,
    )

    hlr_prior = ngmix.priors.FlatPrior(
        0.0001, 1.0e6,
        rng=rng,
    )

    F_prior = ngmix.priors.FlatPrior(
        0.0001, 1.0e9,
        rng=rng,
    )

    if model == 'bdf':
        fracdev_prior = ngmix.priors.TruncatedGaussian(
            0.5,
            0.1,
            -1,
            1,
            rng=rng,
        )
        return ngmix.joint_prior.PriorBDFSep(
            cen_prior,
            g_prior,
            hlr_prior,
            fracdev_prior,
            [F_prior]*nband,
        )
    else:
        return ngmix.joint_prior.PriorSimpleSep(
            cen_prior,
            g_prior,
            hlr_prior,
            [F_prior]*nband,
        )


def get_stamp_guesses_gs(list_of_obs,
                         detband,
                         model,
                         rng,
                         prior=None,
                         guess_from_priors=False):
    """
    get a guess based on metadata in the obs

    T guess is gotten from detband
    """

    nband = len(list_of_obs[0])

    if model == 'bdf':
        npars_per = 6+nband
    else:
        npars_per = 5+nband

    nobj = len(list_of_obs)

    npars_tot = nobj*npars_per
    guess = np.zeros(npars_tot)

    for i, mbo in enumerate(list_of_obs):
        detobslist = mbo[detband]
        detmeta = detobslist.meta

        obs = detobslist[0]

        scale = obs.jacobian.get_scale()
        pos_range = scale*0.1

        if 'Tsky' in detmeta:
            hlr_guess = np.sqrt(detmeta['Tsky'] / 2.0)
        else:
            T = detmeta['T']*scale**2
            hlr_guess = np.sqrt(T / 2.0)

        beg = i*npars_per

        # always close guess for center
        guess[beg+0] = rng.uniform(low=-pos_range, high=pos_range)
        guess[beg+1] = rng.uniform(low=-pos_range, high=pos_range)

        if guess_from_priors:
            pguess = prior.sample()
            # we already guessed the location
            pguess = pguess[2:]
            n = pguess.size
            start = beg+2
            end = start+n
            guess[start:end] = pguess
        else:
            # always arbitrary guess for shape
            guess[beg+2] = rng.uniform(low=-0.05, high=0.05)
            guess[beg+3] = rng.uniform(low=-0.05, high=0.05)

            guess[beg+4] = hlr_guess*(1.0 + rng.uniform(low=-0.05, high=0.05))

            # arbitrary guess for fracdev
            if model == 'bdf':
                guess[beg+5] = rng.uniform(low=0.4, high=0.6)
                flux_start = 6
            else:
                flux_start = 5

            for band, obslist in enumerate(mbo):
                obslist = mbo[band]
                scale = obslist[0].jacobian.scale
                meta = obslist.meta

                # note we take out scale**2 in DES images when
                # loading from MEDS so this isn't needed
                flux = meta['flux']
                flux_guess = flux*(1.0 + rng.uniform(low=-0.05, high=0.05))

                guess[beg+flux_start+band] = flux_guess

    return guess
