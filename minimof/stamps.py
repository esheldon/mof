from __future__ import print_function
import numpy as np
import esutil as eu

class MEDSInterface(object):
    def __init__(self, image, weight, seg, cat):
        self._image_data=dict(
            image=image,
            weight=weight,
            seg=seg,
        )
        self._cat=cat

    def get_cutout(self, iobj, icutout, type='image'):
        """
        Get a single cutout for the indicated entry

        parameters
        ----------
        iobj:
            Index of the object
        icutout:
            Index of the cutout for this object.
        type: string, optional
            Cutout type. Default is 'image'.  Allowed
            values are 'image','weight','seg','bmask'

        returns
        -------
        The cutout image
        """

        if type=='psf':
            return self.get_psf(iobj,icutout)

        self._check_indices(iobj, icutout=icutout)

        if type not in self._image_data:
            raise ValueError("bad cutout type: '%s'" % type)

        im=self._image_data[type]

        c=self._cat
        srow=c['orig_start_row'][iobj,icutout]
        scol=c['orig_start_col'][iobj,icutout]
        erow=c['orig_end_row'][iobj,icutout]
        ecol=c['orig_end_col'][iobj,icutout]

        return im[
            srow:erow,
            scol:ecol,
        ].copy()

    def _check_indices(self, iobj, icutout=None):
        if iobj >= self._cat.size:
            raise ValueError("object index should be within "
                             "[0,%s)" % self._cat.size)

        ncutout=self._cat['ncutout'][iobj]
        if ncutout==0:
            raise ValueError("object %s has no cutouts" % iobj)

        if icutout is not None:
            if icutout >= ncutout:
                raise ValueError("requested cutout index %s for "
                                 "object %s should be in bounds "
                                 "[0,%s)" % (icutout,iobj,ncutout))


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

    def get_meds(self, band):
        """
        get fake MEDS interface to the specified band
        """
        d=self.datalist[band]
        return MEDSInterface(
            d['image'],
            d['weight'],
            self.seg,
            self.cat,
        )

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
        objs, seg = sep.extract(
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



        ncut=2 # need this to make sure array
        new_dt=[
            ('ncutout','i4'),
            ('flux_auto','f4'),
            ('fluxerr_auto','f4'),
            ('flux_radius','f4'),
            ('isoarea_image','f4'),
            ('iso_radius','f4'),
            ('box_size','i4'),
            ('orig_row','f4',ncut),
            ('orig_col','f4',ncut),
            ('orig_start_row','i8',ncut),
            ('orig_start_col','i8',ncut),
            ('orig_end_row','i8',ncut),
            ('orig_end_col','i8',ncut),
            ('cutout_row','f4',ncut),
            ('cutout_col','f4',ncut),
        ]
        cat=eu.numpy_util.add_fields(objs, new_dt)
        cat['ncutout'] = 1
        cat['flux_auto'] = flux_auto
        cat['fluxerr_auto'] = fluxerr_auto
        cat['flux_radius'] = flux_radius


        # use the number of pixels in the seg map as the iso area
        for i in xrange(objs.size):
            w=np.where(seg == (i+1))
            print(i,"found",w[0].size)
            cat['isoarea_image'][i] = w[0].size

        cat['iso_radius'] = np.sqrt(cat['isoarea_image'].clip(min=1)/np.pi)

        RAD_MIN=4 # for box size calculations
        BOX_PADDING=2
        MIN_BOX_SIZE=16
        box_size = (2*cat['iso_radius'].clip(min=RAD_MIN) + BOX_PADDING).astype('i4')
        box_size = box_size.clip(min=MIN_BOX_SIZE,out=box_size)
        wb,=np.where( (box_size % 2) != 0 )
        if wb.size > 0:
            box_size[wb] += 1
        half_box_size = box_size//2

        maxrow,maxcol=self.detim.shape
        cat['box_size'] = 2*cat['iso_radius'] + BOX_PADDING

        cat['orig_row'][:,0] = cat['y']
        cat['orig_col'][:,0] = cat['x']

        orow = cat['orig_row'][:,0].astype('i4')
        ocol = cat['orig_col'][:,0].astype('i4')

        ostart_row = orow - half_box_size + 1
        ostart_col = ocol - half_box_size + 1
        oend_row   = orow + half_box_size + 1 # plus one for slices
        oend_col   = ocol + half_box_size + 1

        ostart_row.clip(min=0, out=ostart_row)
        ostart_col.clip(min=0, out=ostart_col)
        oend_row.clip(max=maxrow, out=oend_row)
        oend_col.clip(max=maxcol, out=oend_col)

        # could result in smaller than box_size above
        cat['orig_start_row'][:,0] = ostart_row
        cat['orig_start_col'][:,0] = ostart_col
        cat['orig_end_row'][:,0] = oend_row
        cat['orig_end_col'][:,0] = oend_col
        cat['cutout_row'][:,0] = cat['orig_row'][:,0] - cat['orig_start_row'][:,0]
        cat['cutout_col'][:,0] = cat['orig_col'][:,0] - cat['orig_start_col'][:,0]


        self.seg=seg
        self.cat=cat

def test():
    import galsim
    import images
    g=galsim.Gaussian(fwhm=1)
    scale=0.263
    im=g.drawImage(scale=scale).array
    noises=[0.001,0.002,0.003]
    dlist=[
        {'image':im+np.random.normal(scale=noises[i],size=im.shape),
         'weight':np.zeros(im.shape)+1.0/noises[i]**2,
         'noise':noises[i],
         'pixel_scale':scale,
        }
        for i in xrange(len(noises))
    ]

    mer=MEDSifier(dlist)
    m=mer.get_meds(0)
    im=m.get_cutout(0,0)
    images.multiview(im)
