"""
assumes the psf is constant across the input larger image
"""
from __future__ import print_function
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
from ngmix.observation import Observation,ObsList,MultiBandObsList,get_mb_obs
from ngmix.gmix import GMixList,MultiBandGMixList

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
            output=MultiBandObsList()
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

            output=ObsList()

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
                psf_obs=Observation(
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


    def get_object_band_pars(self, pars_in, iobj, band):
        nbper=self.nband_pars_per

        pars=np.zeros(nbper)

        # either i*6 or i*7
        beg=0
        # either 5 or 6
        end=0+nbper-1

        ibeg = iobj*self.npars_per
        iend = ibeg+nbper-1

        pars[beg:end] = pars_in[ibeg:iend]

        # now copy the flux
        pars[end] = pars_in[iend+band]

        return pars


    def get_band_pars(self, pars_in, band):
        """
        Get linear pars for the specified band
        """

        pars=self._band_pars
        nbper=self.nband_pars_per

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

class MOFStamps(MOF):
    def __init__(self, list_of_obs, model, **keys):
        """
        list_of_obs is not an ObsList, it is a python list of 
        Observation/ObsList/MultiBandObsList
        """
        #super(LMSimple,self).__init__(obs, model, **keys)

        self._set_all_obs(list_of_obs)
        self._setup_nbrs()
        self.model=model
        self.prior = keys.get('prior',None)

        assert self.prior is not None,"send a prior"
        self.nobj=len(self.list_of_obs)

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

        #self._band_pars=np.zeros(self.nband_pars_per*self.nobj)
        #self._set_fdiff_size()

        self.npars = self.nobj*self.npars_per

        self.lm_pars={}
        self.lm_pars.update(_default_lm_pars)

        lm_pars=keys.get('lm_pars',None)
        if lm_pars is not None:
            self.lm_pars.update(lm_pars)

    def _init_gmix_all(self, pars):
        """
        input pars are in linear space

        initialize the list of lists of gaussian mixtures
        """


        gmix0_lol=[]
        gmix_lol=[]
        for iobj,mbobs in enumerate(self.list_of_obs):

            gmix_all0 = MultiBandGMixList()
            gmix_all  = MultiBandGMixList()

            for band,obs_list in enumerate(mbobs):
                gmix_list0=GMixList()
                gmix_list=GMixList()

                band_pars=self.get_object_band_pars(pars, iobj, band)

                for obs in obs_list:
                    assert obs.has_psf_gmix(),"psfs must be set"

                    # make this make a single model not huge model
                    gm0 = self._make_model(band_pars)
                    psf_gmix=obs.psf.gmix
                    gm=gm0.convolve(psf_gmix)

                    gmix_list0.append(gm0)
                    gmix_list.append(gm)

                gmix_all0.append(gmix_list0)
                gmix_all.append(gmix_list)

            gmix0_lol.append( gmix_all0 )
            gmix_lol.append( gmix_all )

        self.gmix0_lol=gmix0_lol
        self.gmix_lol=gmix_lol

    def _make_model(self, band_pars):
        """
        generate a gaussian mixture
        """
        return GMixModel(band_pars, self.model)

    def _set_all_obs(self, list_of_obs):

        lobs=[]
        for i,o in enumerate(list_of_obs):
            mbo=get_mb_obs(o)
            if i==0:
                self.nband=len(mbo)
            else:
                assert len(mbo)==self.nband,"all obs must have same number of bands"
            lobs.append(mbo)

        self.list_of_obs = lobs

    def _setup_nbrs(self):
        """
        determine which neighboring objects should be
        rendered into each stamp
        """
        for iobj,mbo in enumerate(self.list_of_obs):
            for band in xrange(self.nband):
                band_obslist=mbo[band]

                for icut,obs in enumerate(band_obslist):
                    nbr_data = self._get_nbr_data(obs, iobj, band)
                    obs.meta['nbr_data']=nbr_data
                    print('    obj %d band %d cut %d found %d '
                          'nbrs' % (iobj,band,icut,len(nbr_data)))

    def _get_nbr_data(self, obs, iobj, band):
        """
        TODO trim list to those that we expect to contribute flux
        """

        meta=obs.meta

        nbr_list=[]
        # now look for neighbors that were found in
        # this image
        file_id=obs.meta['file_id']
        for inbr,nbr_mbo in enumerate(self.list_of_obs):
            if inbr==iobj:
                continue

            nbr_band_obslist=nbr_mbo[band]
            for nbr_obs in nbr_band_obslist:
                # only keep the ones in the same image
                # will want to trim later to ones expected to contribute
                # flux
                nbr_meta=nbr_obs.meta
                if nbr_meta['file_id']==file_id:
                    row = nbr_meta['orig_row'] - meta['orig_start_row']
                    col = nbr_meta['orig_col'] - meta['orig_start_col']
                    
                    # this makes a copy
                    jacobian=obs.jacobian
                    jacobian.set_cen(row=row, col=col)

                    nbr_data=dict(
                        index=inbr,
                        jacobian=jacobian,
                    )
                    nbr_list.append(nbr_data)

        return nbr_list


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

def get_stamp_guesses(list_of_obs, detband, model, rng):
    """
    get a guess based on metadata in the obs

    T guess is gotten from detband
    """

    assert model=="bdf"
    nband=len(list_of_obs[0])

    npars_per=6+nband
    nobj=len(list_of_obs)

    npars_tot = nobj*npars_per
    guess = np.zeros(npars_tot)

    for i,mbo in enumerate(list_of_obs):
        detobslist = mbo[detband]
        detmeta=detobslist.meta

        obs=detobslist[0]

        scale=obs.jacobian.get_scale()
        pos_range = scale*0.1

        T=detmeta['T']*scale**2

        beg=i*npars_per

        # always close guess for center
        guess[beg+0] = rng.uniform(low=-pos_range, high=pos_range)
        guess[beg+1] = rng.uniform(low=-pos_range, high=pos_range)

        # always arbitrary guess for shape
        guess[beg+2] = rng.uniform(low=-0.05, high=0.05)
        guess[beg+3] = rng.uniform(low=-0.05, high=0.05)

        guess[beg+4] = T*(1.0 + rng.uniform(low=-0.05, high=0.05))

        # arbitrary guess for fracdev
        guess[beg+5] = rng.uniform(low=0.4,high=0.6)

        for band in xrange(nband):
            obslist=mbo[band]
            meta=obslist.meta

            # note we take out scale**2 in DES images when
            # loading from MEDS so this isn't needed
            flux=meta['flux']*scale**2
            flux_guess=flux*(1.0 + rng.uniform(low=-0.05, high=0.05))

            guess[beg+6+band] = flux_guess

    return guess

def get_mof_prior(list_of_obs, model, rng):
    """
    Not generic, need to let this be configurable
    """
    assert model=="bdf"

    nobj=len(list_of_obs)
    nband=len(list_of_obs[0])

    obs=list_of_obs[0][0][0] 
    cen_sigma=obs.jacobian.get_scale() # a pixel
    cen_prior=ngmix.priors.CenPrior(
        0.0,
        0.0,
        cen_sigma, cen_sigma,
        rng=rng,
    )

    g_prior=ngmix.priors.GPriorBA(
        0.2,
        rng=rng,
    )
    T_prior = ngmix.priors.TwoSidedErf(
        -1.0, 0.1, 1.0e6, 1.0e5,
        rng=rng,
    )

    fracdev_prior = ngmix.priors.Normal(0.0, 0.1, rng=rng)

    F_prior = ngmix.priors.TwoSidedErf(
        -100.0, 1.0, 1.0e9, 1.0e8,
        rng=rng,
    )

    return ngmix.joint_prior.PriorBDFSep(
        cen_prior,
        g_prior,
        T_prior,
        fracdev_prior,
        [F_prior]*nband,
    )


