"""
quick test to see if the algorithm is working
"""
from __future__ import print_function
import numpy
import ngmix

from ngmix.gexceptions import BootPSFFailure, BootGalFailure

from . import minimof

class NBRMaker(dict):
    """
    simulate a central object and neighbors
    """
    def __init__(self, config, rng):
        self.update(config)
        self.rng = rng

    def _get_obj_class(self, model):
        import galsim
        if model=='gauss':
            cls = galsim.Gaussian
        elif model=='exp':
            cls = galsim.Exponential
        elif model=='dev':
            cls = galsim.DeVaucouleurs
        else:
            raise ValueError("bad model: '%s'" % model)

        return cls

    def _get_shift(self, shift_conf):
        assert shift_conf['type'] =='circle'

        theta = self.rng.uniform(low=0.0, high=2*numpy.pi)
        rad = shift_conf['radius']
        shift = rad*numpy.cos(theta),rad*numpy.sin(theta)
        return shift

    def _get_model(self, objconf, shift_conf=None):
        cls = self._get_obj_class(objconf['model'])

        obj = cls(
            half_light_radius = objconf['hlr'],
            flux=objconf['flux'],
        )

        if shift_conf is not None:
            shift = self._get_shift(shift_conf)
            obj = obj.shift(dx=shift[1], dy=shift[0])
        else:
            shift = 0.0, 0.0

        return obj, shift

    def _make_image(self, obj, noise, dims=None):
        """
        get psf realization and image
        """
        imconf=self['image']

        if dims is not None:
            gsim = obj.drawImage(
                nx=dims[1],
                ny=dims[0],
                scale=imconf['scale'],
            )
        else:
            gsim = obj.drawImage(scale=imconf['scale'])

        im = gsim.array

        nim = self.rng.normal(
            scale=noise,
            size=im.shape,
        )
        im += nim

        wt = im*0 + 1.0/noise**2
        return im, wt
 
    def _get_psf(self):
        """
        get psf realization and image
        """

        imconf=self['image']
        psfconf = self['psf']

        psf, psf_shift = self._get_model(psfconf)
        im, wt = self._make_image(psf, psfconf['noise'])

        cen = (numpy.array(im.shape)-1.0)/2.0
        jac = ngmix.DiagonalJacobian(
            row=cen[0],
            col=cen[1],
            scale=imconf['scale'],
        )
        obs = ngmix.Observation(
            im,
            weight=wt,
            jacobian=jac,
        )
        return psf,obs
 
    def __call__(self):
        """
        Copied from Lorena's simple pair maker code
        """
    
        import galsim

        psf, psf_obs = self._get_psf()

        imconf=self['image']
        dims = imconf['dims']
        ccen = (numpy.array(dims)-1.0)/2.0
        objconf = self['objects']

        objs = []
        coords = []

        central_obj, shift = self._get_model(objconf['central'])
        objs.append(central_obj)

        coord = numpy.array(shift) + ccen
        coords.append( coord )

        shift_conf=objconf['nbr_shifts']
        for conf in objconf['nbrs']:
            obj, shift = self._get_model(conf, shift_conf=shift_conf)
            objs.append(obj)

            coord = numpy.array(shift) + ccen
            coords.append( coord )

        objs = galsim.Add(objs)

        shear = self['shear']
        objs  = objs.shear(g1=shear[0], g2=shear[1])    

        objs = galsim.Convolve(objs, psf.withFlux(1.0))

        im, wt = self._make_image(
            objs,
            imconf['noise'],
            dims=dims,
        )

        # now multiple observations based on this image
        all_obslist = []
        for coord in coords:
            jac = ngmix.DiagonalJacobian(
                row=coord[0],
                col=coord[1],
                scale=imconf['scale'],
            )
            obs = ngmix.Observation(
                im.copy(),
                weight=wt.copy(),
                jacobian=jac,
                psf=psf_obs,
            )
            obslist=ngmix.ObsList()
            obslist.append(obs)

            all_obslist.append( obslist)

        return all_obslist

class Tester(object):
    def __init__(self, nsim_conf_name, fit_config_name):
        import nsim

        self.sim_config = {
            'image': {
                'dims': [48,48],
                'noise':10.0,
                'scale':1.0,
            },
            'psf': {
                'model':'gauss',
                'hlr': 1.7,
                'flux': 10000.0,
                'noise': 0.001,
            },
            'objects': {
                'nbr_shifts': {
                    'type':'circle',
                    'radius': 12.0,
                },
                'central': {
                    'model':'gauss',
                    'hlr': 1.7,
                    'flux': 6000.0,
                },
                'nbrs': [
                    {
                        'model':'exp',
                        'hlr':3.4,
                        'flux':8000.0,
                    },
                ]
            },
            'shear': [0.02, 0.00],
        }

        seed = 34123
        rng = numpy.random.RandomState(seed=self['seed'])
        self.sim=NBRMaker(self.sim_config, rng)

        self.fit_config=nsim.files.read_config(fit_config_name)

    def go(self, n=1, doplot=False):

        for i in range(n):
            print("-"*70)
            print("example:",i)
            allobs = self.sim()

            if doplot:
                import images
                images.multiview(allobs[0][0].image,title='orig')
                #if input('hit a key:')=='q':
                #    return

            mm=minimof.MiniMOF(
                self.fit_config['mof'],
                allobs,
                rng=self.sim.rng,
            )
            mm.go()
            res = mm.get_result()
            if not res['converged']:
                print("did not converge")
            else:
                print("converged")
                if doplot:
                    import images
                    corr_obs = mm.get_corrected_obs(0)
                    images.multiview(corr_obs[0].image,title='corr')


            if doplot:
                if input('hit a key:')=='q':
                    return

def test(doplot=False):
    t=Tester('sim-em01nbr','run-em01nbr-mcal-t01')
    res=t.go(100,doplot=doplot)


