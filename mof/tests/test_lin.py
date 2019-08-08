from __future__ import print_function
import ngmix
import numpy as np
import yaml
from ..moftest import Sim
from ..moflib import (
    MOFFlux,
    MOFStamps,
    get_mof_stamps_prior,
    get_stamp_guesses,
)


LM_PARS = {
    'maxfev': 4000,
    'ftol': 1.0e-3,
    'xtol': 1.0e-3,
}

CONF_TEMPLATE = """
# scale of entire group in arcseconds.  Currently this
# is a gaussian sigma
cluster_scale: 2.0
dims: [64,64]
pixel_scale: 0.263

nobj: {nobj}

noise_sigma: 0.04
psf_noise_sigma: 0.0001

nband: 1

pdfs:
    g:
        sigma: 0.2

    hlr:
        type: uniform
        range: [0.5, 0.5]

    F:
        # track the half light radius
        type: uniform
        # range: [1000.0, 1000.0]
        range: [100.0, 100.0]
        # range: [10.0, 10.0]
        # range: [3.0, 3.0]

    disk:
        color: [1.0]

    bulge:
        color: [1.0]

        # g_bulge=g_fac*g_disk
        g_fac:
            type: uniform
            range: [0.5, 0.5]


        fracdev:
            type: uniform
            # range: [0.5, 0.5]
            range: [0.0, 0.0]

        # bulge_hlr=hlr_fac*disk_hlr
        hlr_fac:
            type: uniform
            range: [1.0, 1.0]

        # shift in units of the disk hlr
        bulge_shift: 0.05

fit_model: exp
"""


def _get_conf(nobj):
    conf = CONF_TEMPLATE.format(nobj=nobj)
    return yaml.load(conf)


def _test_nobj(nobj, seed, show=False):
    """
    simulate one object
    """

    detband = 0
    conf = _get_conf(nobj)

    sim = Sim(conf, seed)
    sim.make_obs()
    if show:
        import images
        images.view(sim.obs[0][0].image)
        if raw_input('hit a key: (q to quit): ') == 'q':
            raise KeyboardInterrupt('stopping')

    fitrng = np.random.RandomState(sim.rng.randint(0, 2**15))

    # this runs sextractor
    medser = sim.get_medsifier()

    m2 = medser.get_meds(detband)
    objects = m2.get_cat()
    nobj = len(objects)
    if nobj == 0:
        print('found no objects')
        return

    m = medser.get_multiband_meds()
    list_of_obs = []
    for iobj in range(objects.size):
        mbo = m.get_mbobs(iobj, weight_type='weight')

        for olist in mbo:
            for o in olist:
                o.set_psf(sim.psf_obs)

        list_of_obs.append(mbo)

    # first do a fit
    prior = get_mof_stamps_prior(
        list_of_obs,
        conf['fit_model'],
        fitrng,
    )
    fitter = MOFStamps(
        list_of_obs,
        conf['fit_model'],
        prior=prior,
        lm_pars=LM_PARS,
    )
    guess = get_stamp_guesses(
        list_of_obs,
        detband,
        conf['fit_model'], fitrng,
    )

    for itry in range(2):
        fitter.go(guess)
        res = fitter.get_result()
        if res['flags'] == 0:
            break

    if res['flags'] != 0:
        raise RuntimeError('didnt find a fit')

    print('-'*70)
    print('best fit pars for all objects')
    print('object   pars')
    for i in range(nobj):
        ores = fitter.get_object_result(i)
        s2n = ores['s2n']
        ngmix.print_pars(ores['pars'],
                         front='%d s2n: %.1f pars: ' % (i+1, s2n))

    ffitter = MOFFlux(list_of_obs, conf['fit_model'], res['pars'])
    ffitter.go()

    fres = ffitter.get_result()
    if np.any(fres['flags'] != 0):
        print('didnt find a linear fit')
        return np.zeros(0), np.zeros(0)

    print('-'*70)
    print('flux comparison')
    print('object   flux   flux_err  fflux  fflux_err')

    fluxes = np.zeros(nobj)
    flux_errs = np.zeros(nobj)

    for i in range(nobj):
        ores = fitter.get_object_result(i)
        fres = ffitter.get_object_result(i)
        print(
            'object:', i+1,
            'flux:', ores['flux'][0],
            'flux_err:', ores['flux_err'][0],
            'fflux:', fres['flux'][0],
            'fflux_err:', fres['flux_err'][0],
        )

        fluxes[i] = fres['flux'][0]
        flux_errs[i] = fres['flux_err'][0]

    return fluxes, flux_errs


def test1(show=False):
    _test_nobj(1, 3145, show=show)


def test2(show=False):
    _test_nobj(2, 8712, show=show)


def test_err(ntrial=100, show=False):
    rng = np.random.RandomState(13412)

    all_fluxes = []
    all_flux_errs = []

    for i in range(ntrial):
        seed = rng.randint(0, 2**15)
        fluxes, flux_errs = _test_nobj(1, seed, show=show)
        if fluxes.size > 0:

            imax = fluxes.argmax()
            all_fluxes.append(fluxes[imax])
            all_flux_errs.append(flux_errs[imax])

    all_fluxes = np.array(all_fluxes)
    all_flux_errs = np.array(all_flux_errs)

    import esutil as eu
    actual_std = all_fluxes.std()
    # am, actual_std = eu.stat.sigma_clip(all_fluxes)
    expected_std = all_flux_errs.mean()

    print('measured std:', actual_std)
    print('expected std:', expected_std)

    return all_fluxes, all_flux_errs
