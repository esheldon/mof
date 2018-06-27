"""
assumes the psf is constant across the input larger image
"""
import numpy as np
import ngmix
from ngmix.gmix import GMix, GMixModel
from ngmix.fitting import LMSimple
from ngmix.fitting import run_leastsq
from ngmix.gmix import (
    get_model_num,
    get_model_name,
    get_model_ngauss,
    get_model_npars,
)

# weaker than usual
_default_lm_pars={
    'maxfev':2000,
    'ftol': 1.0e-3,
    'xtol': 1.0e-3,
}


class MOF(LMSimple):
    """
    fit multiple objects simultaneously, but not in postage stamps
    """
    def __init__(self, obs, model, nobj, **keys):
        """
        currently model is same for all objects
        """
        super(LMSimple,self).__init__(obs, model, **keys)

        assert self.prior is not None,"send a prior"
        self.nobj=nobj


        if model=='bdf':
            self.npars_per = 6+self.nband
            self.nband_pars_per=7

            # center1 + center2 + shape + T + fracdev + fluxes for each object
            self.n_prior_pars=self.nobj*(1 + 1 + 1 + 1 + 1 + self.nband)
        else:
            self.npars_per = 5+self.nband
            self.nband_pars_per=6
            self._band_pars=np.zeros(self.nband_pars_per*self.nobj)

            # center1 + center2 + shape + T + fluxes for each object
            self.n_prior_pars=self.nobj*(1 + 1 + 1 + 1 + self.nband)

        self._band_pars=np.zeros(self.nband_pars_per*self.nobj)

        self._set_fdiff_size()

        self.npars = self.nobj*self.npars_per


        self.lm_pars={}
        self.lm_pars.update(_default_lm_pars)

        lm_pars=keys.get('lm_pars',None)
        if lm_pars is not None:
            self.lm_pars.update(lm_pars)


    def go(self, guess):
        """
        Run leastsq and set the result
        """

        guess=np.array(guess,dtype='f8',copy=False)

        # assume 5+nband pars per object
        nobj = guess.size//self.npars_per
        nleft = guess.size % self.npars_per
        if nobj != self.nobj or nleft != 0:
            raise ValueError("bad guess size: %d" % guess.size)

        self._setup_data(guess)

        self._make_lists()

        result = run_leastsq(
            self._calc_fdiff,
            guess,
            self.n_prior_pars,
            **self.lm_pars
        )

        result['model'] = self.model_name
        if result['flags']==0:
            stat_dict=self.get_fit_stats(result['pars'])
            result.update(stat_dict)

        self._result=result

    def get_nobj(self):
        """
        number of input objects we are fitting
        """
        return self.nobj

    def make_corrected_obs(self, index, band=None, obsnum=None, recenter=True):
        """
        get observation(s) for the given object and band
        with all the neighbors subtracted from the image

        parameters
        ----------
        index: number
            The object index.
        band: number, optional
            The optional band.  If not sent, all bands and epochs are returned
            in a MultiBandObsList
        obsnum: number, optional
            If band= is sent, you can also send obsnum to pick a particular
            epoch/observation
        """

        if band is None:
            # get all bands and epochs
            output=ngmix.MultiBandObsList()
            for band in range(self.nband):
                obslist=self.make_corrected_obs(
                    index,
                    band=band,
                    recenter=recenter,
                )
                output.append(obslist)

        elif obsnum is None:
            # band specified, but not the observation, so get all
            # epochs for this band

            output=ngmix.ObsList()

            nepoch = len(self.obs[band])
            for obsnum in range(nepoch):
                obs = self.make_corrected_obs(
                    index,
                    band=band,
                    obsnum=obsnum,
                    recenter=recenter,
                )
                output.append(obs)

        else:
            # band and obsnum specified

            ref_obs = self.obs[band][obsnum]

            image =self.make_corrected_image(index, band=band, obsnum=obsnum)

            if ref_obs.has_psf():
                po=ref_obs.psf
                psf_obs=ngmix.Observation(
                    po.image.copy(),
                    weight=po.weight.copy(),
                    jacobian=po.jacobian.copy(),
                )
                if po.has_gmix():
                    psf_obs.gmix =  po.gmix
            else:
                psf_obs=None

            jacob = ref_obs.jacobian.copy()
            if recenter:

                gm = self.get_gmix(band=band)
                gmi = gm.get_one(index)

                row,col = gmi.get_cen()
                jacob.set_cen(row=row, col=col)

            output = ngmix.Observation(
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

        image = ref_obs.image.copy()
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

        pars=self._band_pars
        nbper=self.nband_pars_per
        nper=self.npars_per

        for i in range(self.nobj):
            # copy cen1,cen2,g1,g2,T
            # or
            # copy cen1,cen2,g1,g2,T,fracdev

            # either i*6 or i*7
            beg=i*nbper
            # either 5 or 6
            end=beg+nbper-1

            ibeg = i*self.npars_per
            iend = ibeg+nbper-1

            pars[beg:end] = pars_in[ibeg:iend]

            # now copy the flux
            #pars[beg+5] = pars_in[ibeg+5+band]
            pars[end] = pars_in[iend+band]

        return pars

    def get_gmix(self, band=0):
        """
        Get a gaussian mixture at the fit parameter set, which
        definition depends on the sub-class

        parameters
        ----------
        band: int, optional
            Band index, default 0
        """
        res=self.get_result()
        band_pars=self.get_band_pars(res['pars'], band)
        return self._make_model(band_pars)

    def get_convolved_gmix(self, band=0, obsnum=0):
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

        gm = self.get_gmix(band=band)
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

        self._model      = get_model_num(model)
        self._model_name = get_model_name(model)

        self._ngauss_per = get_model_ngauss(self._model)
        self._npars_per  = get_model_npars(self._model)

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

        #start = index*self._ngauss_per
        #end = (index+1)*self._ngauss_per
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

        self._pars[:] = pars

        gmall=self.get_data()

        ng=self._ngauss_per
        np=self._npars_per
        for i in range(self._nobj):
            beg=i*ng
            end=(i+1)*ng

            pbeg=i*np
            pend=(i+1)*np

            # should be a reference
            gm = gmall[beg:end]
            gpars = pars[pbeg:pend]

            self._fill_func(
                gm,
                gpars,
            )

    '''
    def make_image(self, band, fast_exp=False):
        """
        render the full model
        """
        obs=self.obs[band][0]

        res=self.get_result()
        if res['flags'] != 0:
            raise RuntimeError("can't render a failure")
        dims=obs.image.shape

        image=numpy.zeros(dims, dtype='f8')

        coords=make_coords(image.shape, obs.jacobian)

        gmall=self.get_data()

        ng=self._ngauss_per
        for i in range(self._nobj):
            beg=i*ng
            end=(i+1)*ng

            # should be a reference
            gm = gmall[beg:end]

            ngmix.render_nb.render(
                gm,
                coords,
                image,
                fast_exp=fast_exp,
            )

        return image
    '''


