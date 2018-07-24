import numpy as np
import esutil as eu

class MEDSifier(object):
    def __init__(self, datalist):
        """
        very simple MEDS maker for images. Assumes the images
        line up, have constant noise, are sky subtracted, 
        and have a simple pixel scale

        The images are added together to make a detection image and sep is run

        parameters
        ----------
        datalist: list
            List of dicts with entries
                image
                noise
                scale
        """
        self.datalist=datalist

        self._set_detim()
        self._run_sep()

    def _set_detim(self):
        dlist=self.datalist
        nim=len(dlist)

        detim=dlist[0]['image'].copy()
        detim *= 0

        vars = np.array( [d['noise']**2 for d in dlist] )
        weights = 1.0/vars
        wsum = weights.sum()
        detnoise = np.sqrt(1/wsum)

        weights /= wsum

        for i,d in enumerate(dlist):
            detim += d['image']*weights[i]

        self.detim=detim
        self.detnoise=detnoise

    def _run_sep(self):
        import sep
        THRESH=1.0 # in sky sigma
        objs, self.segmap = sep.extract(
            self.detim,
            THRESH,
            err=self.detnoise,
            deblend_cont=0.0001,
            segmentation_map=True,
        )

        kronrad, krflag = sep.kron_radius(
            self.detim,
            objs['x'],
            objs['y'],
            objs['a'],
            objs['b'],
            objs['theta'],
            6.0,
        )
        flux_auto, fluxerr_auto, flag_auto = \
            sep.sum_ellipse(
                self.detim,
                objs['x'],
                objs['y'],
                objs['a'],
                objs['b'],
                objs['theta'],
                2.5*kronrad,
                subpix=1,
            )
        objs['flag'] |= krflag

        # should bail now if fails I think

        # what we did in DES, but note threshold above
        # is 1 as opposed to wide survey. deep survey
        # was even lower, 0.8?

        # used half light radius
        PHOT_FLUXFRAC = 0.5

        flux_radius, frflag = sep.flux_radius(
            self.detim,
            objs['x'],
            objs['y'],
            6.*objs['a'],
            PHOT_FLUXFRAC,
            normflux=flux_auto,
            subpix=5,
        )
        objs['flag'] |= frflag  # combine flags into 'flag'

        new_dt=[
            ('flux_auto','f4'),
            ('fluxerr_auto','f4'),
            ('flux_radius','f4'),
        ]

        cat=eu.numpy_util.add_fields(objs, new_dt)
        cat['flux_auto'] = flux_auto
        cat['fluxerr_auto'] = fluxerr_auto
        cat['flux_radius'] = flux_radius

        self.cat=cat

def test():
    import galsim
    g=galsim.Gaussian(fwhm=1)
    scale=0.263
    im=g.drawImage(scale=scale).array
    noises=[0.001,0.002,0.003]
    dlist=[
        {'image':im+np.random.normal(scale=noises[i],size=im.shape),
         'noise':noises[i],
         'pixel_scale':scale,
        }
        for i in range(len(noises))
    ]

    m=MEDSifier(dlist)
