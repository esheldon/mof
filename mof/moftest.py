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

try:
    import galsim
    have_galsim = True
except ImportError:
    have_galsim = False


class Sim(dict):
    def __init__(self, config, seed):
        self.rng = np.random.RandomState(seed)

        self.update(config)

        self.g_pdf = self._make_g_pdf()

        self._make_primary_pdfs()

        self._make_bulge_pdfs()

        if 'knots' in self['pdfs']:
            self._make_knots_pdfs()

        sigma = self['cluster_scale']
        maxrad = 3*sigma

        self.position_pdf = ngmix.priors.TruncatedSimpleGauss2D(
            0.0,
            0.0,
            sigma,
            sigma,
            maxrad,
            rng=self.rng,
        )

    def make_obs(self):
        self._set_bands()
        self._set_psf()
        self._set_objects()
        self._draw_objects()
        self._add_noise()
        self._make_obs()

    def get_obs(self):
        return self.obs

    def get_psf_obs(self):
        return self.psf_obs

    def get_multiband_meds(self):
        medser = self.get_medsifier()
        mm = medser.get_multiband_meds()
        return mm

    def get_medsifier(self):
        from .stamps import MEDSifier
        meds_config = {
            'min_box_size': 32,
            'max_box_size': 128,


            'box_type': 'iso_radius',
            'rad_min': 4,
            'rad_fac': 2,
            'box_padding': 2,
        }

        dlist = []
        for olist in self.obs:
            # assuming only one image per band
            tobs = olist[0]
            wcs = tobs.jacobian.get_galsim_wcs()

            dlist.append(
                dict(
                    image=tobs.image,
                    weight=tobs.weight,
                    wcs=wcs,
                )
            )

        return MEDSifier(dlist, meds_config=meds_config)

    def _make_g_pdf(self):
        c = self['pdfs']['g']
        rng = self.rng
        return ngmix.priors.GPriorBA(c['sigma'], rng=rng)

    def _make_hlr_pdf(self):
        c = self['pdfs']['hlr']
        return self._get_generic_pdf(c)

    def _make_F_pdf(self):
        c = self['pdfs']['F']
        if c['type'] == 'track_hlr':
            return 'track_hlr'
        else:
            return self._get_generic_pdf(c)

    def _make_primary_pdfs(self):
        if 'hlr_flux' in self['pdfs']:
            self.hlr_flux_pdf = self._make_hlr_flux_pdf()
        else:
            self.hlr_pdf = self._make_hlr_pdf()
            self.F_pdf = self._make_F_pdf()

    def _make_hlr_flux_pdf(self):
        from .pdfs import CosmosSampler
        c = self['pdfs']['hlr_flux']
        assert c['type'] == 'cosmos'

        return CosmosSampler(rng=self.rng)

    def _make_bulge_pdfs(self):
        self.bulge_hlr_frac_pdf = self._make_bulge_hlr_frac_pdf()
        self.fracdev_pdf = self._make_fracdev_pdf()

    def _make_bulge_hlr_frac_pdf(self):
        c = self['pdfs']['bulge']['hlr_fac']
        assert c['type'] == 'uniform'

        frng = c['range']
        return ngmix.priors.FlatPrior(frng[0], frng[1], rng=self.rng)

    def _make_fracdev_pdf(self):
        c = self['pdfs']['bulge']['fracdev']
        assert c['type'] == 'uniform'
        frng = c['range']
        return ngmix.priors.FlatPrior(frng[0], frng[1], rng=self.rng)

    def _get_bulge_stats(self):
        c = self['pdfs']['bulge']
        shift_width = c['bulge_shift']

        radial_offset = self.rng.uniform(
            low=0.0,
            high=shift_width,
        )
        theta = self.rng.uniform(low=0, high=np.pi*2)
        offset = (
            radial_offset*np.sin(theta),
            radial_offset*np.cos(theta),
        )

        hlr_fac = self.bulge_hlr_frac_pdf.sample()
        fracdev = self.fracdev_pdf.sample()
        grng = c['g_fac']['range']
        gfac = self.rng.uniform(
            low=grng[0],
            high=grng[1],
        )

        return hlr_fac, fracdev, gfac, offset

    def _get_knots_stats(self, disk_flux):
        c = self['pdfs']['knots']
        nrange = c['num']['range']
        num = self.rng.randint(nrange[0], nrange[1]+1)

        flux = num*c['flux_frac_per_knot']*disk_flux
        return num, flux

    def _make_knots_pdfs(self):
        c = self['pdfs']['knots']['num']
        assert c['type'] == 'uniform'

    def _get_generic_pdf(self, c):
        rng = self.rng

        if c['type'] == 'lognormal':
            pdf = ngmix.priors.LogNormal(
                c['mean'],
                c['sigma'],
                rng=rng,
            )
        elif c['type'] in ['flat', 'uniform']:
            pdf = ngmix.priors.FlatPrior(
                c['range'][0],
                c['range'][1],
                rng=rng,
            )
        else:
            raise ValueError("bad pdf: '%s'" % c['type'])

        limits = c.get('limits', None)
        if limits is None:
            return pdf
        else:
            return ngmix.priors.LimitPDF(pdf, [0.0, 30.0])

    def show(self):
        import images
        images.multiview(self.image)

    def _set_bands(self):
        nband = self.get('nband', None)

        cdisk = self['pdfs']['disk']
        cbulge = self['pdfs']['bulge']
        cknots = self['pdfs'].get('knots', None)

        if nband is None:
            self['nband'] = 1
            cdisk['color'] = [1.0]
            cbulge['color'] = [1.0]
            if cknots is not None:
                cknots['color'] = [1.0]

    def _fit_psf_admom(self, obs):
        Tguess = 4.0*self['pixel_scale']**2
        am = ngmix.admom.run_admom(obs, Tguess)
        return am.get_gmix()

    def _set_psf(self):
        import galsim

        self.psf = galsim.Gaussian(fwhm=0.9)

        kw = {'scale': self['pixel_scale']}
        dims = self.get('psf_dims', None)
        if dims is not None:
            kw['nx'], kw['ny'] = dims[1], dims[0]

        self.psf_im = self.psf.drawImage(**kw).array

        dims = np.array(self.psf_im.shape)
        pcen = (dims-1.0)/2.0
        pjac = ngmix.DiagonalJacobian(
            row=pcen[0],
            col=pcen[1],
            scale=self['pixel_scale']
        )

        self.psf_im += self.rng.normal(
            scale=self['psf_noise_sigma'],
            size=dims,
        )
        psf_wt = np.zeros(dims)+1.0/self['psf_noise_sigma']**2

        self.psf_obs = ngmix.Observation(
            self.psf_im,
            weight=psf_wt,
            jacobian=pjac,
        )

        psf_gmix = self._fit_psf_admom(self.psf_obs)
        self.psf_obs.set_gmix(psf_gmix)

    def _get_hlr_flux(self):
        if 'hlr_flux' in self['pdfs']:
            hlr, flux = self.hlr_flux_pdf.sample()
        else:
            hlr = self.hlr_pdf.sample()

            if self.F_pdf == 'track_hlr':
                flux = hlr**2 * self['pdfs']['F']['factor']
            else:
                flux = self.F_pdf.sample()

        return hlr, flux

    def _get_object(self):

        hlr, flux = self._get_hlr_flux()

        disk_hlr = hlr

        disk_g1, disk_g2 = self.g_pdf.sample2d()

        hlr_fac, fracdev, gfac, bulge_offset = self._get_bulge_stats()

        bulge_hlr = disk_hlr*hlr_fac
        bulge_g1, bulge_g2 = gfac*disk_g1, gfac*disk_g2

        disk_flux = (1-fracdev)*flux
        bulge_flux = fracdev*flux

        disk = galsim.Exponential(
            half_light_radius=disk_hlr,
            flux=disk_flux,
        ).shear(
            g1=disk_g1, g2=disk_g2,
        )

        bulge = galsim.DeVaucouleurs(
            half_light_radius=bulge_hlr,
            flux=bulge_flux,
        ).shear(
            g1=bulge_g1, g2=bulge_g2,
        ).shift(
            dx=bulge_offset[1], dy=bulge_offset[0],
        )

        all_obj = {
            'disk': disk,
            'bulge': bulge,
        }

        if 'knots' in self['pdfs']:
            nknots, knots_flux = self._get_knots_stats(disk_flux)

            knots = galsim.RandomWalk(
                npoints=nknots,
                half_light_radius=disk_hlr,
                flux=knots_flux,
            ).shear(g1=disk_g1, g2=disk_g2)

            all_obj['knots'] = knots

        obj_cen1, obj_cen2 = self.position_pdf.sample()
        all_obj['cen'] = (obj_cen1, obj_cen2)
        return all_obj

    def _set_objects(self):
        self.objlist = []
        for i in range(self['nobj']):

            obj = self._get_object()
            self.objlist.append(obj)

    def _draw_objects(self):
        """
        this is dumb, drawing into the full image when
        we don't need to
        """

        self.imlist = []

        cdisk = self['pdfs']['disk']
        cbulge = self['pdfs']['bulge']
        cknots = self['pdfs'].get('knots', None)

        for band in range(self['nband']):
            objects = []
            for obj_parts in self.objlist:

                disk = obj_parts['disk']*cdisk['color'][band]
                bulge = obj_parts['bulge']*cbulge['color'][band]
                tparts = [disk, bulge]

                if cknots is not None:
                    knots = obj_parts['knots']*cknots['color'][band]
                    tparts.append(knots)

                obj = galsim.Sum(tparts)
                obj = obj.shift(
                    dx=obj_parts['cen'][0],
                    dy=obj_parts['cen'][1],
                )
                objects.append(obj)

            objects = galsim.Sum(objects)

            shear = self.get('shear', None)
            if shear is not None:
                objects = objects.shear(
                    g1=shear[0],
                    g2=shear[1],
                )

            convolved_objects = galsim.Convolve(objects, self.psf)

            kw = {'scale': self['pixel_scale']}

            dims = self.get('dims', None)
            if dims is not None:
                kw['nx'], kw['ny'] = dims[1], dims[0]

            dims = np.array(self.psf_im.shape)
            image = convolved_objects.drawImage(**kw).array
            self.imlist.append(image)

    def _add_noise(self):
        for im in self.imlist:
            noise_image = self.rng.normal(
                scale=self['noise_sigma'],
                size=im.shape,
            )
            im += noise_image

    def _make_obs(self):

        mbobs = ngmix.MultiBandObsList()

        for im in self.imlist:
            jacobian = ngmix.DiagonalJacobian(
                row=0,
                col=0,
                scale=self['pixel_scale']
            )

            wt = np.zeros(im.shape) + 1.0/self['noise_sigma']**2
            obs = ngmix.Observation(
                im,
                weight=wt,
                jacobian=jacobian,
                psf=self.psf_obs,
            )
            olist = ngmix.ObsList()
            olist.append(obs)
            mbobs.append(olist)

        self.obs = mbobs
