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

    @property
    def size(self):
        return self._cat.size

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

    nobj=4
    nband=3
    noises=[0.001,0.002,0.003]
    scale=0.263

    all_band_obj=[]
    for i in xrange(nobj):
        fwhm=np.random.uniform(low=1.0, high=3.0)
        dx,dy=np.random.uniform(low=-5.0, high=5.0, size=2)

        tmp=np.random.uniform(low=0.0, high=1.0)
        if tmp < 0.5:
            colors = np.array([0.5, 1.0, 1.5]) \
                    + np.random.uniform(low=-0.1, high=0.1, size=3)
        else:
            colors = np.array([1.5, 1.0, 0.5]) \
                    + np.random.uniform(low=-0.1, high=0.1, size=3)
        g1,g2=np.random.normal(scale=0.2, size=2).clip(max=0.5)

        obj = galsim.Gaussian(fwhm=fwhm).shift(dx=dx, dy=dy).shear(g1=g1,g2=g2)

        band_objs = [obj.withFlux(colors[j]) for j in xrange(nband)]

        all_band_obj.append( band_objs )

    dlist=[]
    for band in xrange(nband):
        band_objects = [ o[band] for o in all_band_obj ]
        obj = galsim.Sum(band_objects)

        if band==0:
            im = obj.drawImage(scale=scale).array
            dims=im.shape
        else:
            im = obj.drawImage(nx=dims[1], ny=dims[0], scale=scale).array

        im += np.random.normal(scale=noises[band], size=im.shape)
        wt = im*0 + 1.0/noises[band]**2

        dlist.append(
            dict(
                image=im,
                weight=wt,
                noise=noises[band],
                pixel_scale=scale,
            )
        )

    rgb=images.get_color_image(
        #imi*fac,imr*fac,img*fac,
        dlist[2]['image'].transpose(),
        dlist[1]['image'].transpose(),
        dlist[0]['image'].transpose(),
        nonlinear=0.1,
    )
    rgb *= 1.0/rgb.max()
    #images.view(rgb)
 
    mer=MEDSifier(dlist)
    images.view_mosaic( [rgb,mer.seg], dims=[2000,1000])

    mg=mer.get_meds(0)
    mr=mer.get_meds(1)
    mi=mer.get_meds(2)
    nobj=mg.size

    imlist=[]
    for i in xrange(nobj):

        img=mg.get_cutout(i,0)
        imr=mr.get_cutout(i,0)
        imi=mi.get_cutout(i,0)

        rgb=images.get_color_image(
            #imi*fac,imr*fac,img*fac,
            imi,imr,img,
            nonlinear=0.1,
        )
        rgb *= 1.0/rgb.max()
        imlist.append(rgb)

    #images.view(rgb)
    images.view_mosaic(imlist)
