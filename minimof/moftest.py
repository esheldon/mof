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
try:
    import galsim
    have_galsim=True
except ImportError:
    have_galsim=False

class Sim(dict):
    def __init__(self, config, seed):
        self.rng=np.random.RandomState(seed)

        self.update(config)
        #self['dims'] = np.array(self['dims'])

        self.g_pdf = self._make_g_pdf()
        self.hlr_pdf = self._make_hlr_pdf()
        self.F_pdf = self._make_F_pdf()
        if 'bulge+disk' in self['models']:
            self.bulge_hlr_frac_pdf=self._make_bulge_hlr_frac_pdf()
            self.fracdev_pdf=self._make_fracdev_pdf()

        if 'knots' in self['pdfs']:
            self.knots_frac_pdf=self._make_knots_frac_pdf()

        #cen=(self['dims']-1.0)/2.0
        #maxrad=cen[0]-self['dims'][0]/10.0

        sigma=self['cluster_scale']
        maxrad = 3*sigma
        '''
        self.position_pdf=ngmix.priors.SimpleGauss2D(
            0.0,0.0,
            sigma, sigma,
            rng=self.rng,
        )
        '''

        self.position_pdf=ngmix.priors.TruncatedSimpleGauss2D(
            #cen[0], cen[1],
            0.0,0.0,
            sigma, sigma,
            maxrad,
            rng=self.rng,
        )

    def _make_g_pdf(self):
        c=self['pdfs']['g']
        rng=self.rng
        return ngmix.priors.GPriorBA(c['sigma'], rng=rng)

    def _make_hlr_pdf(self):
        c=self['pdfs']['hlr']
        return self._get_generic_pdf(c)

    def _make_F_pdf(self):
        c=self['pdfs']['F']
        if c['type']=='track_hlr':
            return 'track_hlr'
        else:
            return self._get_generic_pdf(c)

    def _make_bulge_hlr_frac_pdf(self):
        c=self['pdfs']['bulge_hlr']
        assert c['fac']['type'] == 'uniform'
        frng=c['fac']['range']
        return ngmix.priors.FlatPrior(frng[0], frng[1], rng=self.rng)

    def _make_fracdev_pdf(self):
        c=self['pdfs']['fracdev']
        assert c['type'] == 'uniform'
        frng=c['range']
        return ngmix.priors.FlatPrior(frng[0], frng[1], rng=self.rng)


    def _make_knots_frac_pdf(self):
        c=self['pdfs']['knots']
        assert c['frac']['type'] == 'uniform'
        frng=c['frac']['range']
        return ngmix.priors.FlatPrior(frng[0], frng[1], rng=self.rng)


    def _get_generic_pdf(self, c):
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
        if limits is None:
            return pdf
        else:
            return ngmix.priors.LimitPDF(pdf, [0.0, 30.0])

    def make_obs(self):
        self._set_psf()
        self._set_objects()
        self._draw_objects()
        self._add_noise()
        self._make_obs()

    def get_obs(self):
        return self.obs

    def get_image(self):
        return self.image

    def show(self):
        images.multiview(self.image)

    def _fit_psf_admom(self, obs):
        Tguess=4.0*self['pixel_scale']**2
        am=ngmix.admom.run_admom(obs, Tguess)
        return am.get_gmix()

    def _set_psf(self):
        import galsim

        self.psf = galsim.Gaussian(fwhm=0.9)

        kw={'scale':self['pixel_scale']}
        dims=self.get('psf_dims',None)
        if dims is not None:
            kw['nx'],kw['ny'] = dims[1],dims[0]

        self.psf_im = self.psf.drawImage(**kw).array

        dims = np.array(self.psf_im.shape)
        pcen=(dims-1.0)/2.0
        pjac = ngmix.DiagonalJacobian(
            row=pcen[0],
            col=pcen[1],
            scale=self['pixel_scale']
        )


        self.psf_im += self.rng.normal(
            scale=self['psf_noise_sigma'],
            size=dims,
        )
        psf_wt=np.zeros(dims)+1.0/self['psf_noise_sigma']**2

        self.psf_obs = ngmix.Observation(
            self.psf_im,
            weight=psf_wt,
            jacobian=pjac,
        )

        psf_gmix=self._fit_psf_admom(self.psf_obs)
        self.psf_obs.set_gmix(psf_gmix)

    def _get_bulgedisk_object(self):
        disk_g1,disk_g2 = self.g_pdf.sample2d()
        bulge_g1,bulge_g2 = self.g_pdf.sample2d()

        disk_hlr = self.hlr_pdf.sample()
        bulge_shift_width = disk_hlr*self['pdfs']['bulge_shift']

        bulge_dx, bulge_dy = self.rng.uniform(
            low=-bulge_shift_width,
            high=bulge_shift_width,
            size=2,
        )


        bulge_hlr = disk_hlr*self.bulge_hlr_frac_pdf.sample()

        if self.F_pdf=='track_hlr':
            flux = disk_hlr**2 *self['pdfs']['F']['factor']
        else:
            flux = self.F_pdf.sample()

        fracdev = self.fracdev_pdf.sample()

        total_disk_flux = (1-fracdev)*flux
        bulge_flux = fracdev*flux

        if hasattr(self,'knots_frac_pdf'):
            knots_frac = self.knots_frac_pdf.sample()
            smooth_frac = 1 - knots_frac

            smooth_flux = smooth_frac*total_disk_flux
            knots_flux = knots_frac*total_disk_flux

            smooth_disk = galsim.Exponential(
                half_light_radius=disk_hlr,
                flux=smooth_flux,
            )

            #knots = galsim.RandomWalk(
            #    npoints=self['pdfs']['knots']['num'],
            #    profile=smooth_disk.withFlux(knots_flux),
            #)
            knots = galsim.RandomWalk(
                npoints=self['pdfs']['knots']['num'],
                half_light_radius=disk_hlr,
                flux=knots_flux,
            )

            disk = galsim.Add(smooth_disk, knots)
        else:
            disk = galsim.Exponential(
                half_light_radius=disk_hlr,
                flux=total_disk_flux,
            )

        disk = disk.shear(g1=disk_g1, g2=disk_g2)

        bulge=galsim.DeVaucouleurs(
            half_light_radius=bulge_hlr,
            flux=bulge_flux,
        ).shift(dx=bulge_dx, dy=bulge_dy)

        return galsim.Sum(bulge, disk)


    def _get_object(self):
        """
        draw a random model
        """

        i=self.rng.randint(0, len(self['models']))
        modname=self['models'][i]

        obj_cen1, obj_cen2 = self.position_pdf.sample()

        if modname=='bulge+disk':
            obj = self._get_bulgedisk_object()
        else:
            obj = self._get_simple_object()

        obj = obj.shift(
            dx=obj_cen2,
            dy=obj_cen1,
        )
        return obj

    def _set_objects(self):
        rng=self.rng
        obj_pars = []

        self.objlist=[]
        for i in range(self['nobj']):

            obj = self._get_object()
            self.objlist.append(obj)

    def _draw_objects(self):
        """
        this is dumb, drawing into the full image when
        we don't need to
        """

        objects = galsim.Sum(self.objlist)

        shear=self.get('shear',None)
        if shear is not None:
            objects = objects.shear(
                g1=shear[0],
                g2=shear[1],
            )

        convolved_objects = galsim.Convolve(objects, self.psf)

        kw={'scale':self['pixel_scale']}

        dims = self.get('dims',None)
        if dims is not None:
            kw['nx'],kw['ny'] = dims[1],dims[0]

        dims = np.array(self.psf_im.shape)
        self.image0 = convolved_objects.drawImage(**kw).array

        #import images
        #images.multiview(self.image0)
        #stop

    def _add_noise(self):
        noise_image = self.rng.normal(
            scale=self['noise_sigma'],
            size=self.image0.shape,
        )
        self.image = self.image0 + noise_image

    def _make_obs(self):

        self.jacobian = ngmix.DiagonalJacobian(
            row=0,
            col=0,
            scale=self['pixel_scale']
        )

        wt=np.zeros(self.image.shape) + 1.0/self['noise_sigma']**2
        self.obs = ngmix.Observation(
            self.image,
            weight=wt,
            jacobian=self.jacobian,
            psf=self.psf_obs,
        )


