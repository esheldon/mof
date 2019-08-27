"""
assumes the psf is constant across the input larger image
todo:
    need to deal with odd fits in the mof when subtracting, somehow
    they are getting g >= 1

    maybe it actually failed and we aren't detecting that?
"""
from __future__ import print_function
import numpy as np
from numpy import dot
from numba import njit
import ngmix
from ngmix.gmix import GMix, GMixModel, GMixBDF
from ngmix.fitting import LMSimple
from ngmix.fitting import run_leastsq
from ngmix.gmix import (
    get_model_num,
    get_model_name,
    get_model_ngauss,
    get_model_npars,
)
from ngmix.observation import (
    Observation, ObsList, MultiBandObsList,
    get_mb_obs,
)
from ngmix.gmix import GMixList, MultiBandGMixList
from ngmix.gexceptions import GMixRangeError
from ngmix.priors import LOWVAL
from . import priors

# weaker than usual
DEFAULT_LM_PARS = {
    'ftol': 1.0e-5,
    'xtol': 1.0e-5,
}


class MOF(LMSimple):
    """
    fit multiple objects simultaneously, but not in postage stamps
    """
    def __init__(self, obs, model, nobj, **keys):
        """
        currently model is same for all objects
        """
        super(LMSimple, self).__init__(obs, model, **keys)

        assert self.prior is not None, 'send a prior'
        self.nobj = nobj

        if model == 'bd':
            self.npars_per = 7+self.nband
            self.nband_pars_per = 8

            # center1 + center2 + shape + T + log10Trat +fracdev + fluxes
            self.n_prior_pars = self.nobj*(1 + 1 + 1 + 1 + 1 + 1 + self.nband)

        elif model == 'bdf':
            self.npars_per = 6+self.nband
            self.nband_pars_per = 7

            # center1 + center2 + shape + T + fracdev + fluxes for each object
            self.n_prior_pars = self.nobj*(1 + 1 + 1 + 1 + 1 + self.nband)
        else:
            self.npars_per = 5+self.nband
            self.nband_pars_per = 6
            self._band_pars = np.zeros(self.nband_pars_per*self.nobj)

            # center1 + center2 + shape + T + fluxes for each object
            self.n_prior_pars = self.nobj*(1 + 1 + 1 + 1 + self.nband)

        self._band_pars = np.zeros(self.nband_pars_per*self.nobj)

        self._set_fdiff_size()

        self.npars = self.nobj*self.npars_per

        self.lm_pars = {}
        self.lm_pars.update(DEFAULT_LM_PARS)

        lm_pars = keys.get('lm_pars', None)
        if lm_pars is not None:
            self.lm_pars.update(lm_pars)

        if 'maxfev' not in self.lm_pars:
            # default in leastsq is 100*(self.npars+1)
            self.lm_pars['maxfev'] = 300*(self.npars+1)

    def go(self, guess):
        """
        Run leastsq and set the result
        """

        guess = np.array(guess, dtype='f8', copy=False)

        # assume 5+nband pars per object
        nobj = guess.size//self.npars_per
        nleft = guess.size % self.npars_per
        if nobj != self.nobj or nleft != 0:
            raise ValueError("bad guess size: %d" % guess.size)

        self._setup_data(guess)

        self._make_lists()

        bounds = self._get_bounds(nobj)

        result = run_leastsq(
            self._calc_fdiff,
            guess,
            self.n_prior_pars,
            bounds=bounds,
            **self.lm_pars
        )

        result['model'] = self.model_name
        if result['flags'] == 0:
            stat_dict = self.get_fit_stats(result['pars'])
            result.update(stat_dict)

        self._result = result

    def _get_bounds(self, nobj):
        """
        get bounds on parameters
        """
        bounds = None
        if self.prior is not None:
            if (hasattr(self.prior, 'bounds') and
                    self.prior.bounds is not None):
                bounds = self.prior.bounds
                bounds = bounds*nobj

        return bounds

    def get_nobj(self):
        """
        number of input objects we are fitting
        """
        return self.nobj

    def make_corrected_obs(self,
                           index=None,
                           band=None,
                           obsnum=None,
                           recenter=True):
        """
        get observation(s) for the given object and band
        with all the neighbors subtracted from the image

        parameters
        ----------
        index: number, optional
            The object index. If not sent, a list of all corrected
            observations is returned
        band: number, optional
            The optional band.  If not sent, all bands and epochs are returned
            in a MultiBandObsList
        obsnum: number, optional
            If band= is sent, you can also send obsnum to pick a particular
            epoch/observation
        """

        if index is None:
            mbobs_list = []
            for index in range(self.nobj):
                mbobs = self.make_corrected_obs(
                    index=index,
                    band=band,
                    obsnum=obsnum,
                    recenter=recenter,
                )
                mbobs_list.append(mbobs)
            return mbobs_list

        if band is None:
            # get all bands and epochs
            output = MultiBandObsList()
            for band in range(self.nband):
                obslist = self.make_corrected_obs(
                    index=index,
                    band=band,
                    recenter=recenter,
                )
                output.append(obslist)

        elif obsnum is None:
            # band specified, but not the observation, so get all
            # epochs for this band

            output = ObsList()

            nepoch = len(self.obs[band])
            for obsnum in range(nepoch):
                obs = self.make_corrected_obs(
                    index=index,
                    band=band,
                    obsnum=obsnum,
                    recenter=recenter,
                )
                output.append(obs)

        else:
            # band and obsnum specified

            ref_obs = self.obs[band][obsnum]

            image = self.make_corrected_image(index, band=band, obsnum=obsnum)

            if ref_obs.has_psf():
                po = ref_obs.psf
                psf_obs = Observation(
                    po.image.copy(),
                    weight=po.weight.copy(),
                    jacobian=po.jacobian.copy(),
                )
                if po.has_gmix():
                    psf_obs.gmix = po.gmix
            else:
                psf_obs = None

            # this makes a copy
            jacob = ref_obs.jacobian
            if recenter:

                gm = self.get_gmix(band=band)
                gmi = gm.get_one(index)

                # this is v,u
                v, u = gmi.get_cen()
                row, col = jacob.get_rowcol(v, u)
                jacob.set_cen(row=row, col=col)

            output = Observation(
                image,
                weight=ref_obs.weight.copy(),
                jacobian=jacob,
                psf=psf_obs,
            )

        return output

    def make_corrected_image(self, index, band=0, obsnum=0):
        """
        get an observation for the given object and band
        with all the neighbors subtracted from the image
        """

        ref_obs = self.obs[band][obsnum]
        image = ref_obs.image.copy()

        if self.nobj == 1:
            return image

        model_image = self.make_image(band=band, obsnum=obsnum)

        # now remove the object of interest from the model
        # image
        gm = self.get_gmix(band=band)
        gmi = gm.get_one(index)

        if ref_obs.has_psf():
            psf = ref_obs.psf.gmix
            gmi = gmi.convolve(psf)

        iimage = gmi.make_image(model_image.shape, jacobian=ref_obs.jacobian)
        model_image -= iimage

        image -= model_image

        return image

    def make_image(self, band=0, obsnum=0):
        """
        make an image for the given band and observation number
        """
        gm = self.get_convolved_gmix(band=band, obsnum=obsnum)
        obs = self.obs[band][obsnum]
        return gm.make_image(obs.image.shape, jacobian=obs.jacobian)

    def get_band_pars(self, pars_in, band):
        """
        Get linear pars for the specified band
        """

        pars = self._band_pars
        nbper = self.nband_pars_per

        for i in range(self.nobj):
            # copy cen1,cen2,g1,g2,T
            # or
            # copy cen1,cen2,g1,g2,T,fracdev

            # either i*6 or i*7
            beg = i*nbper
            # either 5 or 6
            end = beg+nbper-1

            ibeg = i*self.npars_per
            iend = ibeg+nbper-1

            pars[beg:end] = pars_in[ibeg:iend]

            # now copy the flux
            # pars[beg+5] = pars_in[ibeg+5+band]
            pars[end] = pars_in[iend+band]

        return pars

    def get_gmix(self, band=0, pars=None):
        """
        Get a gaussian mixture at the fit parameter set, which
        definition depends on the sub-class

        parameters
        ----------
        band: int, optional
            Band index, default 0
        """
        if pars is None:
            res = self.get_result()
            pars = res['pars']
        band_pars = self.get_band_pars(pars, band)
        return self._make_model(band_pars)

    def get_convolved_gmix(self, band=0, obsnum=0, pars=None):
        """
        get a gaussian mixture at the fit parameters, convolved by the psf if
        fitting a pre-convolved model

        parameters
        ----------
        band: int, optional
            Band index, default 0
        obsnum: int, optional
            Number of observation for the given band,
            default 0
        """

        gm = self.get_gmix(band=band, pars=pars)
        obs = self.obs[band][obsnum]
        if obs.has_psf_gmix():
            gm = gm.convolve(obs.psf.gmix)

        return gm

    def _make_model(self, band_pars):
        """
        generate a gaussian mixture with the right number of
        components
        """
        return GMixModelMulti(band_pars, self.model)

    def get_object_pars(self, pars_in, iobj):
        """
        extract parameters for the given object
        """
        ibeg = iobj*self.npars_per
        iend = (iobj+1)*self.npars_per

        return pars_in[ibeg:iend].copy()

    def get_object_cov(self, cov_in, iobj):
        """
        extract covariance for the given object
        """
        ibeg = iobj*self.npars_per
        iend = (iobj+1)*self.npars_per

        return cov_in[ibeg:iend, ibeg:iend].copy()

    def get_object_psf_stats(self, *args, **kw):
        """
        we have a single psf for full fitter so this needs to be cached only
        once
        """
        if not hasattr(self, '_psf_stats'):
            g1sum = 0.0
            g2sum = 0.0
            Tsum = 0.0
            wsum = 0.0

            for obslist in self.obs:
                for obs in obslist:
                    twsum = obs.weight.sum()
                    wsum += twsum

                    tg1, tg2, tT = obs.psf.gmix.get_g1g2T()

                    g1sum += tg1*twsum
                    g2sum += tg2*twsum
                    Tsum += tT*twsum

            g1 = g1sum/wsum
            g2 = g2sum/wsum
            T = Tsum/wsum

            self._psf_stats = {
                'g': [g1, g2],
                'T': T,
            }

        stats = {}
        stats.update(self._psf_stats)
        return stats

    def get_result(self):
        """
        full result for all objects
        """
        return self._result

    def get_result_list(self):
        """
        get results split up for each object
        """
        if not hasattr(self, '_result_list'):
            self._make_result_list()

        return self._result_list

    def _make_result_list(self):
        """
        get fit statistics for each object separately
        """
        reslist = []
        for i in range(self.nobj):
            res = self.get_object_result(i)
            reslist.append(res)

        self._result_list = reslist

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
        res['T'] = res['pars'][4]
        res['T_err'] = np.sqrt(res['pars_cov'][4, 4])
        res['T_ratio'] = res['T']/res['psf_T']

        if self.model_name == 'bd':
            res['logTratio'] = res['pars'][5]
            res['logTratio_err'] = np.sqrt(res['pars_cov'][5, 5])
            res['fracdev'] = res['pars'][6]
            res['fracdev_err'] = np.sqrt(res['pars_cov'][6, 6])
            flux_start = 7
        elif self.model_name == 'bdf':
            res['fracdev'] = res['pars'][5]
            res['fracdev_err'] = np.sqrt(res['pars_cov'][5, 5])
            flux_start = 6
        else:
            flux_start = 5

        res['flux'] = res['pars'][flux_start:]
        res['flux_cov'] = res['pars_cov'][flux_start:, flux_start:]
        res['flux_err'] = np.sqrt(np.diag(res['flux_cov']))

        return res

    def get_object_s2n(self, i):
        """
        we don't have a stamp over which to integrate, so we use
        the total flux s/n
        """
        allpars = self._result['pars']
        allpars_cov = self._result['pars_cov']
        pars = self.get_object_pars(allpars, i)
        pars_cov = self.get_object_cov(allpars_cov, i)

        if self.model_name == 'bd':
            flux_start = 7
        elif self.model_name == 'bdf':
            flux_start = 6
        else:
            flux_start = 5

        flux = pars[flux_start:]
        flux_cov = pars_cov[flux_start:, flux_start:]

        fones = np.ones(flux.size)
        flux_cov_inv = np.linalg.inv(flux_cov)

        fvar_inv = flux_cov_inv.sum()
        if fvar_inv <= 0.0:
            flux_s2n = -9999.0
        else:
            fvar = 1/fvar_inv
            flux_avg = dot(fones, dot(flux_cov_inv, flux))*fvar
            flux_avg_err = np.sqrt(fvar)

            flux_s2n = flux_avg/flux_avg_err

        return flux_s2n


class MOFStamps(MOF):
    def __init__(self, list_of_obs, model, **keys):
        """
        list_of_obs is not an ObsList, it is a python list of
        Observation/ObsList/MultiBandObsList
        """

        self._set_all_obs(list_of_obs)
        self._setup_nbrs()
        self.model = model

        self.model = ngmix.gmix.get_model_num(model)
        self.model_name = ngmix.gmix.get_model_name(self.model)

        self.prior = keys.get('prior', None)

        assert self.prior is not None, "send a prior"
        self.nobj = len(self.list_of_obs)
        self._set_totpix()

        if model == 'bd':
            self.npars_per = 7+self.nband
            self.nband_pars_per = 8

            # center1 + center2 + shape + T + logTratio + fracdev + fluxes for
            # each object
            self.n_prior_pars = self.nobj*(1 + 1 + 1 + 1 + 1 + 1 + self.nband)

        elif model == 'bdf':
            self.npars_per = 6+self.nband
            self.nband_pars_per = 7

            # center1 + center2 + shape + T + fracdev + fluxes for each object
            self.n_prior_pars = self.nobj*(1 + 1 + 1 + 1 + 1 + self.nband)
        else:
            self.npars_per = 5+self.nband
            self.nband_pars_per = 6
            self._band_pars = np.zeros(self.nband_pars_per*self.nobj)

            # center1 + center2 + shape + T + fluxes for each object
            self.n_prior_pars = self.nobj*(1 + 1 + 1 + 1 + self.nband)

        self._set_fdiff_size()

        self.npars = self.nobj*self.npars_per

        self.lm_pars = {}
        self.lm_pars.update(DEFAULT_LM_PARS)

        lm_pars = keys.get('lm_pars', None)
        if lm_pars is not None:
            self.lm_pars.update(lm_pars)

        if 'maxfev' not in self.lm_pars:
            # default in leastsq is 100*(self.npars+1)
            self.lm_pars['maxfev'] = 300*(self.npars+1)

    def go(self, guess):
        """
        Run leastsq and set the result
        """

        guess = np.array(guess, dtype='f8', copy=False)

        # assume 5+nband pars per object
        nobj = guess.size//self.npars_per
        nleft = guess.size % self.npars_per
        if nobj != self.nobj or nleft != 0:
            raise ValueError("bad guess size: %d" % guess.size)

        self._setup_data(guess)

        bounds = self._get_bounds(nobj)

        result = run_leastsq(
            self._calc_fdiff,
            guess,
            self.n_prior_pars,
            bounds=bounds,
            **self.lm_pars
        )

        result['model'] = self.model_name
        if result['flags'] == 0:
            stat_dict = self.get_fit_stats(result['pars'])
            result.update(stat_dict)

        self._result = result

    def get_result_averaged_shapes(self):
        """
        not doing anything smart with the rest
        of the parameters yet, just copying first
        object
        """
        res0 = self.get_object_result(0)
        if self.nobj == 1:
            return res0

        res = self.get_result()

        pars = res['pars']
        pcov = res['pars_cov']

        # do separately for now
        g1 = np.zeros(self.nobj)
        g1cov = np.zeros((self.nobj, self.nobj))
        g2 = np.zeros(self.nobj)
        g2cov = np.zeros((self.nobj, self.nobj))

        nper = self.npars_per
        for i in range(self.nobj):

            ig1p = i*nper + 2
            ig2p = i*nper + 2 + 1

            g1[i] = pars[ig1p]
            g2[i] = pars[ig2p]

            for j in range(self.nobj):

                jg1p = j*nper + 2
                jg2p = j*nper + 2 + 1

                g1cov[i, j] = pcov[ig1p, jg1p]
                g2cov[i, j] = pcov[ig2p, jg2p]

        g1cov_inv = np.linalg.inv(g1cov)
        g2cov_inv = np.linalg.inv(g2cov)
        gones = np.ones(g1.size)

        g1var = 1/g1cov_inv.sum()
        g2var = 1/g2cov_inv.sum()
        g1avg = dot(gones, dot(g1cov_inv, g1))*g1var
        g2avg = dot(gones, dot(g2cov_inv, g2))*g2var

        res0['g'] = g1avg, g2avg
        res0['g_cov'] = np.diag([g1var, g2var])

        return res0

    def get_object_s2n(self, i):
        """
        get the s/n for the given object.  This uses just the model
        to calculate the s/n, but does use the full weight map
        """
        s2n_sum = 0.0
        mbobs = self.list_of_obs[i]
        for band, obslist in enumerate(mbobs):
            for obsnum, obs in enumerate(obslist):
                gm = self.get_convolved_gmix(i, band=band, obsnum=obsnum)
                s2n_sum += gm.get_model_s2n_sum(obs)

        return np.sqrt(s2n_sum)

    def get_object_psf_stats(self, i):
        """
        each object can have different psf for stamps version
        """
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

    def make_corrected_obs(self, index=None, band=None, obsnum=None):
        """
        get observation(s) for the given object and band
        with all the neighbors subtracted from the image

        parameters
        ----------
        index: number, optional
            The object index. If not sent, a list of all corrected
            observations is returned
        index: number
            The object index.
        band: number, optional
            The optional band.  If not sent, all bands and epochs are returned
            in a MultiBandObsList
        obsnum: number, optional
            If band= is sent, you can also send obsnum to pick a particular
            epoch/observation
        """

        if index is None:
            mbobs_list = []
            for index in range(self.nobj):
                mbobs = self.make_corrected_obs(
                    index=index,
                    band=band,
                    obsnum=obsnum,
                )
                mbobs_list.append(mbobs)
            return mbobs_list

        if band is None:
            # get all bands and epochs
            output = MultiBandObsList()
            for band in range(self.nband):
                obslist = self.make_corrected_obs(
                    index=index,
                    band=band,
                )
                output.append(obslist)

            all_pars = self._result['pars']
            pars = self.get_object_pars(all_pars, index)
            output.meta['fit_pars'] = pars

        elif obsnum is None:
            # band specified, but not the observation, so get all
            # epochs for this band

            output = ObsList()

            obslist = self.list_of_obs[index][band]
            nepoch = len(obslist)
            for obsnum in range(nepoch):
                obs = self.make_corrected_obs(
                    index=index,
                    band=band,
                    obsnum=obsnum,
                )
                output.append(obs)

        else:
            # band and obsnum specified

            ref_obs = self.list_of_obs[index][band][obsnum]

            image = self.make_corrected_image(index, band=band, obsnum=obsnum)

            output = ref_obs.copy()
            output.image = image

        return output

    def make_corrected_image(self, index, band=0, obsnum=0):
        """
        get an observation for the given object and band
        with all the neighbors subtracted from the image
        """
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

    def get_object_band_pars(self, pars_in, iobj, band):
        nbper = self.nband_pars_per

        pars = np.zeros(nbper)

        # either i*6 or i*7
        beg = 0
        # either 5 or 6
        end = 0+nbper-1

        ibeg = iobj*self.npars_per
        iend = ibeg+nbper-1

        pars[beg:end] = pars_in[ibeg:iend]

        # now copy the flux
        pars[end] = pars_in[iend+band]

        return pars

    def _set_totpix(self):
        """
        Make sure the data are consistent.
        """

        totpix = 0
        for mbobs in self.list_of_obs:
            for obs_list in mbobs:
                for obs in obs_list:
                    totpix += obs.pixels.size

        self.totpix = totpix

    def get_fit_stats(self, pars):
        return {}

    def _calc_fdiff(self, pars):
        """
        vector with (model-data)/error.

        The npars elements contain -ln(prior)
        """

        # we cannot keep sending existing array into leastsq, don't know why
        fdiff = np.zeros(self.fdiff_size)
        start = 0

        try:

            for iobj, mbo in enumerate(self.list_of_obs):
                # fill priors and get new start
                objpars = self.get_object_pars(pars, iobj)
                start = self._fill_priors(objpars, fdiff, start)

                for band, obslist in enumerate(mbo):
                    for obs in obslist:

                        meta = obs.meta
                        pixels = obs.pixels

                        gm0 = meta['gmix0']
                        gm = meta['gmix']
                        psf_gmix = obs.psf.gmix

                        tpars = self.get_object_band_pars(
                            pars,
                            iobj,
                            band,
                        )

                        self._update_model(tpars,
                                           gm0, gm, psf_gmix,
                                           pixels, fdiff, start)

                        # now also do same for neighbors. We can re-use
                        # the gmixes
                        for nbr in meta['nbr_data']:
                            tnbr_pars = self.get_object_band_pars(
                                pars,
                                nbr['index'],
                                band,
                            )
                            # the current pars [v,u,..] are relative to
                            # fiducial position.  we need to add these to
                            # the fiducial for the rendering within
                            # the stamp of the central
                            tnbr_pars[0] += nbr['v0']
                            tnbr_pars[1] += nbr['u0']
                            self._update_model(tnbr_pars,
                                               gm0, gm, psf_gmix,
                                               pixels, fdiff, start)

                        # convert model values to fdiff
                        ngmix.fitting_nb.finish_fdiff(
                            obs._pixels,
                            fdiff,
                            start,
                        )

                        start += pixels.size

        except GMixRangeError:
            fdiff[:] = LOWVAL

        return fdiff

    def _fill_priors(self, pars, fdiff, start):
        """
        same prior for every object
        """

        nprior = self.prior.fill_fdiff(pars, fdiff[start:])

        return start+nprior

    def _update_model(self, pars, gm0, gm, psf_gmix,
                      pixels, model_array, start):
        gm0._fill(pars)
        ngmix.gmix_nb.gmix_convolve_fill(
            gm._data,
            gm0._data,
            psf_gmix._data,
        )

        ngmix.fitting_nb.update_model_array(
            gm._data,
            pixels,
            model_array,
            start,
        )

    def make_image(self, index, band=0, obsnum=0, include_nbrs=False):
        """
        make an image for the given band and observation number
        """
        gm = self.get_convolved_gmix(index, band=band, obsnum=obsnum)
        obs = self.list_of_obs[index][band][obsnum]
        im = gm.make_image(obs.image.shape, jacobian=obs.jacobian)

        if include_nbrs:
            pars = self.get_result()['pars']
            for nbr in obs.meta['nbr_data']:
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
                ngm0 = self._make_model(nbr_pars)
                ngm = ngm0.convolve(obs.psf.gmix)
                nim = ngm.make_image(obs.image.shape, jacobian=obs.jacobian)
                im += nim

        return im

    def get_convolved_gmix(self, index, band=0, obsnum=0, pars=None):
        """
        get the psf-convolved gmix for the specified object, band, obsnum
        """

        gm0 = self.get_gmix(index, band=band, pars=pars)

        obs = self.list_of_obs[index][band][obsnum]
        psf_gmix = obs.psf.gmix
        return gm0.convolve(psf_gmix)

    def get_gmix(self, index, band=0, pars=None):
        """
        get the pre-psf gmix for the specified object, band, obsnum
        """
        if pars is None:
            res = self.get_result()
            pars = res['pars']

        tpars = self.get_object_band_pars(
            pars,
            index,
            band,
        )

        gm0 = self._make_model(tpars)
        return gm0

    def _init_gmix_all(self, pars):
        """
        input pars are in linear space

        initialize the list of lists of gaussian mixtures
        """

        for iobj, mbobs in enumerate(self.list_of_obs):
            for band, obs_list in enumerate(mbobs):
                band_pars = self.get_object_band_pars(pars, iobj, band)
                for obs in obs_list:
                    assert obs.has_psf_gmix(), "psfs must be set"

                    # make this make a single model not huge model
                    gm0 = self._make_model(band_pars)
                    psf_gmix = obs.psf.gmix
                    gm = gm0.convolve(psf_gmix)

                    obs.meta['gmix0'] = gm0
                    obs.meta['gmix'] = gm

    def _init_gmix_all_old(self, pars):
        """
        input pars are in linear space

        initialize the list of lists of gaussian mixtures
        """

        gmix0_lol = []
        gmix_lol = []
        for iobj, mbobs in enumerate(self.list_of_obs):

            gmix_all0 = MultiBandGMixList()
            gmix_all = MultiBandGMixList()

            for band, obs_list in enumerate(mbobs):
                gmix_list0 = GMixList()
                gmix_list = GMixList()

                band_pars = self.get_object_band_pars(pars, iobj, band)

                for obs in obs_list:
                    assert obs.has_psf_gmix(), "psfs must be set"

                    # make this make a single model not huge model
                    gm0 = self._make_model(band_pars)
                    psf_gmix = obs.psf.gmix
                    gm = gm0.convolve(psf_gmix)

                    gmix_list0.append(gm0)
                    gmix_list.append(gm)

                gmix_all0.append(gmix_list0)
                gmix_all.append(gmix_list)

            gmix0_lol.append(gmix_all0)
            gmix_lol.append(gmix_all)

        self.gmix0_lol = gmix0_lol
        self.gmix_lol = gmix_lol

    def _make_model(self, band_pars):
        """
        generate a gaussian mixture
        """

        if self.model_name == 'bdf':
            return GMixBDF(pars=band_pars)
        else:
            return GMixModel(band_pars, self.model)

    def _set_all_obs(self, list_of_obs):

        lobs = []
        for i, o in enumerate(list_of_obs):
            mbo = get_mb_obs(o)
            if i == 0:
                self.nband = len(mbo)
            else:
                assert len(mbo) == self.nband,\
                    "all obs must have same number of bands"
            lobs.append(mbo)

        self.list_of_obs = lobs

    def _setup_nbrs(self):
        """
        determine which neighboring objects should be
        rendered into each stamp
        """
        for iobj, mbo in enumerate(self.list_of_obs):
            for band in range(self.nband):
                band_obslist = mbo[band]

                for icut, obs in enumerate(band_obslist):
                    nbr_data = self._get_nbr_data(obs, iobj, band)
                    obs.meta['nbr_data'] = nbr_data

    def _get_nbr_data(self, obs, iobj, band):
        """
        TODO trim list to those that we expect to contribute flux
        """

        jacobian = obs.jacobian
        meta = obs.meta

        nbr_list = []
        # now look for neighbors that were found in
        # this image
        file_id = obs.meta['file_id']
        for inbr, nbr_mbo in enumerate(self.list_of_obs):
            if inbr == iobj:
                continue

            nbr_band_obslist = nbr_mbo[band]
            for nbr_obs in nbr_band_obslist:
                # only keep the ones in the same image
                # will want to trim later to ones expected to contribute
                # flux
                nbr_meta = nbr_obs.meta
                if nbr_meta['file_id'] == file_id:
                    # row,col within the postage stamp of the central object
                    row = nbr_meta['orig_row'] - meta['orig_start_row']
                    col = nbr_meta['orig_col'] - meta['orig_start_col']

                    # note pars are [v,u,g1,g2,...]
                    v, u = jacobian(row, col)
                    nbr_data = {
                        'v0': v,
                        'u0': u,
                        'index': inbr,
                    }
                    nbr_list.append(nbr_data)

        return nbr_list


class MOFFlux(MOFStamps):
    def __init__(self, list_of_obs, model, pars, flags=None):
        self._set_all_obs(list_of_obs)
        self._setup_nbrs()
        self.model = model

        self.model = ngmix.gmix.get_model_num(model)
        self.model_name = ngmix.gmix.get_model_name(self.model)

        self.prior = None
        self.n_prior_pars = 0

        self.nobj = len(self.list_of_obs)
        self._set_totpix()

        # these are not the returned number of pars, but the
        # input pars
        if model == 'bd':
            self.npars_per = 7+self.nband
            self.nband_pars_per = 8
        elif model == 'bdf':
            self.npars_per = 6+self.nband
            self.nband_pars_per = 7
        else:
            self.npars_per = 5+self.nband
            self.nband_pars_per = 6

        self.npars = self.nobj*self.npars_per

        self._set_input_pars(pars, flags)

    def get_gmix(self, index, band=0, pars=None):
        """
        get the pre-psf gmix for the specified object, band, obsnum
        """

        if pars is not None:
            gm0 = self._make_model(pars)
        else:
            gm0 = self.list_of_obs[index][band][0].meta['gmix0'].copy()

        return gm0

    def _set_input_pars(self, pars, flags):

        pars = np.array(pars, dtype='f8', copy=False)

        nobj = pars.shape[0]
        if nobj != self.nobj:
            raise ValueError('got nobj %d, expected %d' % (nobj, self.nobj))

        npars_per_input = pars.shape[1]

        if npars_per_input > self.npars_per:
            # extract a subset, the flux we might lose doesn't matter
            pars = pars[:, :self.npars_per]

        elif npars_per_input < self.npars_per:

            # we need to expand the pars to have the right shape
            parsold = pars
            pars = np.zeros((pars.shape[0], self.npars_per))
            pars[:, :npars_per_input] = parsold[:, :]
            # value doesn't matter
            pars[:, npars_per_input:] = 1.0

        # self._input_pars = pars.ravel()
        self._input_pars = pars

        if flags is not None:
            if flags.size != pars.shape[0]:
                m = ('incompatible flags shape %s and pars '
                     'shape %s' % (flags.shape, pars.shape))
                raise ValueError(m)

        self._input_flags = flags

        self._setup_data(self._input_pars)

    def go(self):
        """
        the pars should be from a different run with full fitting.

        These will have fluxes but we will not use them
        """

        flux = np.zeros((self.nobj, self.nband)) - 9999.0
        flux_err = np.zeros((self.nobj, self.nband)) + 9999.0
        flags = np.zeros((self.nobj, self.nband))

        for band in range(self.nband):

            try:
                band_res = self._get_lin_flux_band(band)

                flux[:, band] = band_res['flux']
                flux_err[:, band] = band_res['flux_err']

            except GMixRangeError as err:
                print(str(err))
                flags[:, band] = 1
            except np.linalg.LinAlgError as err:
                print(str(err))
                flags[:, band] = 2

        self._result = {
            'model': self.model_name,
            'flux': flux,
            'flux_err': flux_err,
            'flags': flags,
        }

    def _get_lin_flux_band(self, band):

        rim = np.zeros(self.totpix[band])

        flux = np.zeros(self.nobj)
        flux_err = np.zeros(self.nobj)

        for ipass in range(2):
            start = 0
            starts = np.zeros(self.nobj, dtype='i4')
            model_data = np.zeros((self.totpix[band], self.nobj))

            for iobj, mbo in enumerate(self.list_of_obs):

                obslist = mbo[band]
                for obs in obslist:

                    meta = obs.meta
                    pixels = obs.pixels

                    rim[start:start+pixels.size] = pixels['val']*pixels['ierr']
                    start += pixels.size

                    gm0 = meta['gmix0']
                    gm = meta['gmix']
                    psf_gmix = obs.psf.gmix

                    tpars = self.get_object_band_pars(
                        self._input_pars,
                        iobj,
                        band,
                    )

                    # templates always have flux 1
                    if ipass == 0:
                        tpars[-1] = 1.0
                    else:
                        tpars[-1] = flux[iobj]

                    self._set_weighted_model(
                        tpars,
                        gm0, gm, psf_gmix,
                        pixels,
                        model_data[:, iobj],
                        starts[iobj],
                    )
                    starts[iobj] += pixels.size

                    # now also do same for neighbors. We can re-use
                    # the gmixes
                    for nbr in meta['nbr_data']:
                        tnbr_pars = self.get_object_band_pars(
                            self._input_pars,
                            nbr['index'],
                            band,
                        )
                        if ipass == 0:
                            tnbr_pars[-1] = 1.0
                        else:
                            tnbr_pars[-1] = flux[nbr['index']]

                        # the current pars [v,u,..] are relative to
                        # fiducial position.  we need to add these to
                        # the fiducial for the rendering within
                        # the stamp of the central
                        tnbr_pars[0] += nbr['v0']
                        tnbr_pars[1] += nbr['u0']
                        self._set_weighted_model(
                            tnbr_pars,
                            gm0, gm, psf_gmix,
                            pixels,
                            model_data[:, nbr['index']],
                            starts[nbr['index']],
                        )
                        starts[nbr['index']] += pixels.size

            if ipass == 0:
                flux[:], resid, rank, s = np.linalg.lstsq(
                    model_data,
                    rim,
                    rcond=None,
                )
                model_data_nonorm = model_data
            else:

                # this version gives the same error for 1 or N objects,
                # whereas the true error increases somewhat with N
                for i in range(self.nobj):

                    imodel = model_data[:, iobj]
                    imodel_nonorm = model_data_nonorm[:, iobj]
                    msq_sum = (imodel_nonorm**2).sum()

                    subim = rim.copy()
                    for j in range(self.nobj):
                        if j != i:
                            subim -= model_data[:, j]

                    tchi2 = ((subim-imodel)**2).sum()

                    arg = tchi2/msq_sum/(rim.size-self.nobj)
                    if arg > 0:
                        flux_err[i] = np.sqrt(arg)

        return {
            'flux': flux,
            'flux_err': flux_err,
        }

    def _set_weighted_model(self, pars, gm0, gm, psf_gmix,
                            pixels, model_array, start):

        for i in range(2):
            # ngmix.print_pars(pars, front='pars: ')
            gm0._fill(pars)
            # print('gm0 T:',gm0.get_T())
            # print('psf_gm:',psf_gmix.get_g1g2T())
            ngmix.gmix_nb.gmix_convolve_fill(
                gm._data,
                gm0._data,
                psf_gmix._data,
            )
            # print('det:', gm._data['det'])
            # print('T:', gm._data['irr'] + gm._data['icc'])

            try:
                set_weighted_model(
                    gm._data,
                    pixels,
                    model_array,
                    start,
                )
                break
            except GMixRangeError as err:
                print(str(err))
                print('trying zero size')
                parsold = pars
                pars = parsold.copy()
                pars[4] = 0.0

    def get_object_band_pars(self, input_pars, iobj, band):
        # ngmix.print_pars(input_pars[iobj], front='input pars all: ')
        if self._input_flags is None or self._input_flags[iobj] == 0:
            nbper = self.nband_pars_per

            pars = np.zeros(nbper)

            # either i*6 or i*7
            beg = 0
            # either 5 or 6
            end = 0+nbper-1

            # ibeg = iobj*self.npars_per
            # iend = ibeg+nbper-1

            # pars[beg:end] = input_pars[ibeg:iend]
            pars[beg:end] = input_pars[iobj, beg:end]

            # now copy the flux
            pars[end] = input_pars[iobj, end+band]
        else:
            print('    filling a star for missing pars')
            pars = np.zeros(self.nband_pars_per)
            pars[4] = 1.0e-5

        return pars

    def get_object_result(self, i):
        """
        get a result dict for a single object
        """
        pres = self.get_object_psf_stats(i)

        all_res = self._result

        res = {}

        res['nband'] = self.nband
        res['psf_g'] = pres['g']
        res['psf_T'] = pres['T']

        res['nfev'] = 1
        # res['s2n'] = self.get_object_s2n(i)

        res['flux'] = all_res['flux'][i, :]
        res['flux_err'] = all_res['flux_err'][i, :]
        res['pars'] = res['flux'].copy()
        res['pars_err'] = res['flux_err'].copy()
        res['pars_cov'] = np.diag(res['flux_err']**2)

        return res

    def _set_totpix(self):
        """
        Make sure the data are consistent.
        """

        totpix = np.zeros(self.nband, dtype='i4')
        for mbobs in self.list_of_obs:
            for band, obs_list in enumerate(mbobs):
                for obs in obs_list:
                    totpix[band] += obs.pixels.size

        self.totpix = totpix


class MOFFluxOld(MOFStamps):
    def __init__(self, list_of_obs, model, **keys):
        self._set_all_obs(list_of_obs)
        self._setup_nbrs()
        self.model = model

        self.model = ngmix.gmix.get_model_num(model)
        self.model_name = ngmix.gmix.get_model_name(self.model)

        self.prior = None
        self.n_prior_pars = 0

        self.nobj = len(self.list_of_obs)
        self._set_totpix()

        self.npars_per = self.nband
        if model == 'bd':
            self.nband_pars_per_full = 8
        elif model == 'bdf':
            self.nband_pars_per_full = 7
        else:
            self.nband_pars_per_full = 6

        self.npars = self.nobj*self.npars_per

        self._set_fdiff_size()

        self.lm_pars = {}
        self.lm_pars.update(DEFAULT_LM_PARS)

        lm_pars = keys.get('lm_pars', None)
        if lm_pars is not None:
            self.lm_pars.update(lm_pars)

        if 'maxfev' not in self.lm_pars:
            # default in leastsq is 100*(self.npars+1)
            self.lm_pars['maxfev'] = 300*(self.npars+1)

    def get_object_band_flux(self, flux_pars, iobj, band):
        """
        get the input pars plus the flux
        """

        ind = iobj*self.npars_per + band
        flux = flux_pars[ind]
        return flux

    def get_object_band_pars(self, flux_pars, iobj, band):
        """
        get the input pars plus the flux
        """

        flux = self.get_object_band_flux(flux_pars, iobj, band)

        meta = self.list_of_obs[iobj].meta
        flags = meta['input_flags']
        if flags == 0:
            pars = meta['input_model_pars']
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


# TODO move to ngmix
class GMixModelMulti(GMix):
    """
    A two-dimensional gaussian mixture created from a set of model parameters
    for multiple objects

    Inherits from the more general GMix class, and all its methods.

    parameters
    ----------
    pars: array-like
        Parameter array. The number of elements will depend
        on the model type, the total number being nobj*npars_model
    model: string or gmix type
        e.g. 'exp' or GMIX_EXP
    """
    def __init__(self, pars, model):

        self._model = get_model_num(model)
        self._model_name = get_model_name(model)

        if self._model_name == 'bdf':
            self._TdByTe = 1.0

        self._ngauss_per = get_model_ngauss(self._model)
        self._npars_per = get_model_npars(self._model)

        np = len(pars)
        self._nobj = np//self._npars_per
        if (np % self._npars_per) != 0:
            raise ValueError("bad number of pars: %s" % np)

        self._npars = self._nobj*self._npars_per
        self._ngauss = self._nobj*self._ngauss_per

        self.reset()

        self._set_fill_func()
        self.fill(pars)

    def get_nobj(self):
        """
        number of objects represented
        """
        return self._nobj

    def copy(self):
        """
        Get a new GMix with the same parameters
        """
        gmix = GMixModelMulti(self._pars, self._model_name)
        return gmix

    def get_one(self, index):
        """
        extract the mixture for one of the component
        models
        """
        if index > (self._nobj-1):
            raise ValueError("index %d out of "
                             "bounds [0,%d]" % (index, self._nobj-1))

        start = index*self._npars_per
        end = (index+1)*self._npars_per

        pars = self._pars[start:end]
        return GMixModel(pars, self._model_name)

    def set_cen(self, row, col):
        """
        Move the mixture to a new center

        set pars as well
        """
        raise NotImplementedError("would only make sense if multiple "
                                  "rows and cols sent")

    def _fill(self, pars):
        """
        Fill in the gaussian mixture with new parameters, without
        error checking

        parameters
        ----------
        pars: ndarray or sequence
            The parameters
        """

        if self._model_name == 'bdf':
            dobdf = True
        else:
            dobdf = False

        self._pars[:] = pars

        gmall = self.get_data()

        ng = self._ngauss_per
        np = self._npars_per
        for i in range(self._nobj):
            beg = i*ng
            end = (i+1)*ng

            pbeg = i*np
            pend = (i+1)*np

            # should be a reference
            gm = gmall[beg:end]
            gpars = pars[pbeg:pend]

            if dobdf:
                self._fill_func(
                    gm,
                    gpars,
                    self._TdByTe,
                )
            else:
                self._fill_func(
                    gm,
                    gpars,
                )


def get_full_image_guesses(objects,
                           nband,
                           jacobian,
                           model,
                           rng,
                           Tguess=None,
                           prior=None,
                           guess_from_priors=False):

    if model == 'bdf':
        npars_per = 6+nband
    else:
        npars_per = 5+nband

    scale = jacobian.get_scale()
    pos_range = scale*0.1

    nobj = len(objects)

    npars_tot = nobj*npars_per

    if guess_from_priors:
        guess = prior.sample()
        assert guess.size == npars_tot

        # now just fix the centers because we want tighter guesses than
        # the prior
        for i in range(nobj):
            row = objects['y'][i]
            col = objects['x'][i]

            v, u = jacobian(row, col)

            beg = i*npars_per

            # always close guess for center
            guess[beg+0] = v + rng.uniform(low=-pos_range, high=pos_range)
            guess[beg+1] = u + rng.uniform(low=-pos_range, high=pos_range)

    else:
        guess = np.zeros(npars_tot)
        for i in range(nobj):
            row = objects['y'][i]
            col = objects['x'][i]

            v, u = jacobian(row, col)

            if Tguess is not None:
                T = Tguess
            else:
                T = scale**2 * (objects['x2'][i] + objects['y2'][i])

            flux = scale**2 * objects['flux'][i]

            beg = i*npars_per

            # always close guess for center
            guess[beg+0] = v + rng.uniform(low=-pos_range, high=pos_range)
            guess[beg+1] = u + rng.uniform(low=-pos_range, high=pos_range)

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

                guess[beg+4] = T*(1.0 + rng.uniform(low=-0.05, high=0.05))

                if model == 'bdf':
                    # arbitrary guess for fracdev
                    guess[beg+5] = rng.uniform(low=0.4, high=0.6)
                    flux_start = 6
                else:
                    flux_start = 5

                # arbitrary guess for fracdev
                for band in range(nband):
                    flux_guess = flux*(1.0 + rng.uniform(low=-0.05, high=0.05))
                    guess[beg+flux_start+band] = flux_guess

    return guess


def get_stamp_guesses(list_of_obs,
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
            T = detmeta['Tsky']
        else:
            # not good if bands have different scales
            T = detmeta['T']*scale**2

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

            guess[beg+4] = T*(1.0 + rng.uniform(low=-0.05, high=0.05))

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
                flux = meta['flux']*scale**2
                flux_guess = flux*(1.0 + rng.uniform(low=-0.05, high=0.05))

                guess[beg+flux_start+band] = flux_guess

    return guess


def get_mof_full_image_prior(objects, nband, jacobian, model, rng):
    """
    Note a single jacobian is being sent.  for multi-band this
    is the same as assuming they are all on the same coordinate system.

    assuming all images have the
    prior for N objects.  The priors are the same for
    structural parameters, the only difference being the
    centers
    """

    nobj = len(objects)

    cen_priors = []

    cen_sigma = jacobian.get_scale()  # a pixel
    for i in range(nobj):
        row = objects['y'][i]
        col = objects['x'][i]

        v, u = jacobian(row, col)
        p = ngmix.priors.CenPrior(
            v,
            u,
            cen_sigma, cen_sigma,
            rng=rng,
        )
        cen_priors.append(p)

    g_prior = ngmix.priors.GPriorBA(
        0.2,
        rng=rng,
    )

    T_prior = ngmix.priors.TwoSidedErf(
        -1.0, 0.1, 1.0e6, 1.0e5,
        rng=rng,
    )

    F_prior = ngmix.priors.TwoSidedErf(
        -100.0, 1.0, 1.0e9, 1.0e8,
        rng=rng,
    )

    if model == 'bdf':
        fracdev_prior = ngmix.priors.Normal(
            0.5,
            0.1,
            bounds=[0, 1],
            rng=rng,
        )

        return priors.PriorBDFSepMulti(
            cen_priors,
            g_prior,
            T_prior,
            fracdev_prior,
            [F_prior]*nband,
        )
    else:
        return priors.PriorSimpleSepMulti(
            cen_priors,
            g_prior,
            T_prior,
            [F_prior]*nband,
        )


def get_mof_stamps_prior(list_of_obs, model, rng):
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
    T_prior = ngmix.priors.TwoSidedErf(
        -1.0, 0.1, 1.0e6, 1.0e5,
        rng=rng,
    )

    F_prior = ngmix.priors.TwoSidedErf(
        -100.0, 1.0, 1.0e9, 1.0e8,
        rng=rng,
    )

    if model == 'bdf':
        fracdev_prior = ngmix.priors.Normal(
            0.5,
            0.1,
            bounds=[0, 1],
            rng=rng,
        )

        return ngmix.joint_prior.PriorBDFSep(
            cen_prior,
            g_prior,
            T_prior,
            fracdev_prior,
            [F_prior]*nband,
        )
    else:
        return ngmix.joint_prior.PriorSimpleSep(
            cen_prior,
            g_prior,
            T_prior,
            [F_prior]*nband,
        )


@njit
def set_weighted_model(gmix, pixels, arr, start):
    """
    fill 1d array

    parameters
    ----------
    gmix: gaussian mixture
        See gmix.py
    pixels: array if pixel structs
        u,v,val,ierr
    arr: array
        Array to fill
    """

    if gmix['norm_set'][0] == 0:
        ngmix.gmix_nb.gmix_set_norms(gmix)

    n_pixels = pixels.shape[0]
    for ipixel in range(n_pixels):
        pixel = pixels[ipixel]

        model_val = ngmix.gmix_nb.gmix_eval_pixel(gmix, pixel)
        arr[start+ipixel] = model_val*pixel['ierr']
