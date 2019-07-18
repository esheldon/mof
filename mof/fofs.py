from __future__ import print_function, division
import os
import copy
import numpy as np
import meds


class MEDSNbrs(object):
    """
    Gets nbrs of any postage stamp in the MEDS.

    A nbr is defined as any stamp which overlaps the stamp under consideration
    given a buffer or is in the seg map. See the code below.

    Options:
        buff_type - how to compute buffer length for stamp overlap
            'min': minimum of two stamps
            'max': max of two stamps
            'tot': sum of two stamps

        buff_frac - fraction by whch to multiply the buffer

        maxsize_to_replace - postage stamp size to replace with maxsize
        maxsize - size ot use instead of maxsize_to_replace to compute overlap

        check_seg - use object's seg map to get nbrs in addition to postage
        stamp overlap
    """

    def __init__(self, meds_list, conf, cat=None):

        if isinstance(meds_list, meds.MEDS):
            meds_list = [meds_list]

        self.meds_list = meds_list
        self.conf = conf

        self._init_bounds()

    def _init_bounds(self):
        if self.conf['method'] == 'radius':
            return self._init_bounds_by_radius()
        else:
            return self._init_bounds_by_stamps()

    def _init_bounds_by_radius(self):

        radius_name = self.conf['radius_column']

        self.left = {}
        self.right = {}
        self.top = {}
        self.bot = {}
        self.sze = {}

        min_radius = self.conf.get('min_radius', None)
        if min_radius is None:
            min_radius = 1.0

        max_radius = self.conf.get('max_radius', None)
        if max_radius is None:
            max_radius = np.inf

        for band, m in enumerate(self.meds_list):

            r = m[radius_name].copy()

            r *= self.conf['radius_mult']

            r.clip(min=min_radius, max=max_radius, out=r)

            r += self.conf['padding']

            rowcen = m['orig_row'][:, 0]
            colcen = m['orig_col'][:, 0]

            # factor of 2 because this should be a diameter as it is used later
            diameter = r*2
            self.sze[band] = diameter

            self.left[band] = rowcen - r
            self.right[band] = rowcen + r
            self.bot[band] = colcen - r
            self.top[band] = colcen + r

    def _init_bounds_by_stamps(self):
        self.left = {}
        self.right = {}
        self.top = {}
        self.bot = {}
        self.sze = {}

        for band, m in enumerate(self.meds_list):
            # expand the stamps and get edges
            dsize = self.conf['new_maxsize']-self.conf['maxsize_to_replace']
            dsize = dsize//2
            self.sze[band] = m['box_size'].copy()
            self.left[band] = m['orig_start_row'][:, 0].copy()
            self.right[band] = m['orig_start_row'][:, 0].copy()
            self.bot[band] = m['orig_start_col'][:, 0].copy()
            self.top[band] = m['orig_start_col'][:, 0].copy()

            q, = np.where(self.sze[band] == self.conf['maxsize_to_replace'])
            if q.size > 0:
                self.sze[band][q[:]] = self.conf['new_maxsize']
                self.left[band][q[:]] -= dsize
                self.bot[band][q[:]] -= dsize

            self.right[band] += self.sze[band]
            self.top[band] += self.sze[band]

    def get_nbrs(self, verbose=True):
        nbrs_data = []
        dtype = [('number', 'i8'), ('nbr_number', 'i8')]

        for mindex in range(self.meds_list[0].size):
            nbrs = []
            for band, m in enumerate(self.meds_list):
                # make sure MEDS lists have the same objects!
                self._check_ids(m, mindex)

                # add on the nbrs
                nbrs.extend(list(self.check_mindex(mindex, band)))

            # only keep unique nbrs
            nbrs = np.unique(np.array(nbrs))

            # add to final list
            for nbr in nbrs:
                nbrs_data.append((m['number'][mindex], nbr))

        # return array sorted by number
        nbrs_data = np.array(nbrs_data, dtype=dtype)
        i = np.argsort(nbrs_data['number'])
        nbrs_data = nbrs_data[i]

        return nbrs_data

    def _check_ids(self, m, mindex):
        assert m['number'][mindex] == self.meds_list[0]['number'][mindex]
        assert m['id'][mindex] == self.meds_list[0]['id'][mindex]
        assert m['number'][mindex] == mindex+1

    def check_mindex(self, mindex, band):
        m = self.meds_list[band]

        # check that current gal has OK stamp, or return bad crap
        if (m['orig_start_row'][mindex, 0] == -9999 or
                m['orig_start_col'][mindex, 0] == -9999):
            nbr_numbers = np.array([-1], dtype=int)
            return nbr_numbers

        # get the nbrs from two sources
        # 1) intersection of postage stamps
        # 2) seg map vals
        nbr_numbers = []

        # box intersection test and exclude yourself
        # use buffer of 1/4 of smaller of pair
        # sze is a diameter

        if self.conf['method'] == 'radius':
            # we don't add any additional buffering when calculating
            # overlap by radius
            buff = self.sze[band]*0
        else:
            buff = self.sze[band].copy()
            if self.conf['buff_type'] == 'min':
                q, = np.where(buff[mindex] < buff)
                if len(q) > 0:
                    buff[q[:]] = buff[mindex]
            elif self.conf['buff_type'] == 'max':
                q, = np.where(buff[mindex] > buff)
                if len(q) > 0:
                    buff[q[:]] = buff[mindex]
            elif self.conf['buff_type'] == 'tot':
                buff = buff[mindex] + buff
            else:
                assert False, \
                    "buff_type '%s' not supported!" % self.conf['buff_type']

            buff = buff*self.conf['buff_frac']

        q, = np.where(
            (~((self.left[band][mindex] > self.right[band]-buff)
               | (self.right[band][mindex] < self.left[band]+buff)
               | (self.top[band][mindex] < self.bot[band]+buff)
               | (self.bot[band][mindex] > self.top[band]-buff)))
            &
            (m['number'][mindex] != m['number'])
            & (m['orig_start_row'][:, 0] != -9999)
            & (m['orig_start_col'][:, 0] != -9999)
        )

        if len(q) > 0:
            nbr_numbers.extend(list(m['number'][q]))

        # check coadd seg maps
        if self.conf['check_seg']:
            try:
                segmap = m.get_cutout(mindex, 0, type='seg')
                q = np.where((segmap > 0) & (segmap != m['number'][mindex]))
                if len(q) > 0:
                    nbr_numbers.extend(list(np.unique(segmap[q])))
            except:  # noqa
                pass

        # cut weird crap
        if len(nbr_numbers) > 0:
            nbr_numbers = np.array(nbr_numbers, dtype=int)
            nbr_numbers = np.unique(nbr_numbers)
            inds = nbr_numbers-1
            q, = np.where(
                (m['orig_start_row'][inds, 0] != -9999)
                &
                (m['orig_start_col'][inds, 0] != -9999)
            )
            if len(q) > 0:
                nbr_numbers = list(nbr_numbers[q])
            else:
                nbr_numbers = []

        # if have stuff return unique else return -1
        if len(nbr_numbers) == 0:
            nbr_numbers = np.array([-1], dtype=int)
        else:
            nbr_numbers = np.array(nbr_numbers, dtype=int)
            nbr_numbers = np.unique(nbr_numbers)

        return nbr_numbers


class NbrsFoF(object):
    def __init__(self, nbrs_data):
        self.nbrs_data = nbrs_data
        self.Nobj = len(np.unique(nbrs_data['number']))

        # records fofid of entry
        self.linked = np.zeros(self.Nobj, dtype='i8')
        self.fofs = {}

        self._fof_data = None

    def get_fofs(self, verbose=True):
        self._make_fofs(verbose=verbose)
        return self._fof_data

    def _make_fofs(self, verbose=True):
        # init
        self._init_fofs()

        for i in range(self.Nobj):
            self._link_fof(i)

        for fofid, k in enumerate(self.fofs):
            inds = np.array(list(self.fofs[k]), dtype=int)
            self.linked[inds[:]] = fofid
        self.fofs = {}

        self._make_fof_data()

    def _link_fof(self, mind):
        # get nbrs for this object
        nbrs = set(self._get_nbrs_index(mind))

        # always make a base fof
        if self.linked[mind] == -1:
            fofid = copy.copy(mind)
            self.fofs[fofid] = set([mind])
            self.linked[mind] = fofid
        else:
            fofid = copy.copy(self.linked[mind])

        # loop through nbrs
        for nbr in nbrs:
            if self.linked[nbr] == -1 or self.linked[nbr] == fofid:
                # not linked so add to current
                self.fofs[fofid].add(nbr)
                self.linked[nbr] = fofid
            else:
                # join!
                self.fofs[self.linked[nbr]] |= self.fofs[fofid]
                del self.fofs[fofid]
                fofid = copy.copy(self.linked[nbr])
                inds = np.array(list(self.fofs[fofid]), dtype=int)
                self.linked[inds[:]] = fofid

    def _make_fof_data(self):
        self._fof_data = []
        for i in range(self.Nobj):
            self._fof_data.append((self.linked[i], i+1))
        self._fof_data = np.array(
            self._fof_data,
            dtype=[('fofid', 'i8'), ('number', 'i8')],
        )
        i = np.argsort(self._fof_data['number'])
        self._fof_data = self._fof_data[i]
        assert np.all(self._fof_data['fofid'] >= 0)

    def _init_fofs(self):
        self.linked[:] = -1
        self.fofs = {}

    def _get_nbrs_index(self, mind):
        q, = np.where(
            (self.nbrs_data['number'] == mind+1)
            &
            (self.nbrs_data['nbr_number'] > 0)
        )
        if len(q) > 0:
            return list(self.nbrs_data['nbr_number'][q]-1)
        else:
            return []


def plot_fofs(m,
              fof,
              orig_dims=None,
              type='dot',
              fof_type='dot',
              fof_size=1,
              minsize=2,
              show=False,
              width=1000,
              plotfile=None):
    """
    make an ra,dec plot of the FOF groups

    Only groups with at least two members ares shown
    """
    import random
    try:
        import biggles
        import esutil as eu
        have_biggles = True
    except ImportError:
        have_biggles = False

    if not have_biggles:
        print("skipping FOF plot because biggles is not "
              "available")
        return

    x = m['orig_col'][:, 0]
    y = m['orig_row'][:, 0]

    hd = eu.stat.histogram(fof['fofid'], more=True)
    wlarge, = np.where(hd['hist'] >= minsize)
    ngroup = wlarge.size
    if ngroup > 0:
        colors = rainbow(ngroup)
        random.shuffle(colors)
    else:
        colors = None

    print("unique groups >= 2:", wlarge.size)
    print("largest fof:", hd['hist'].max())

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    if orig_dims is not None:
        xmin, xmax = 0, orig_dims[1]
        ymin, ymax = 0, orig_dims[0]
        xrng = [xmin, xmax]
        yrng = [ymin, ymax]
        aratio = (ymax-ymin)/(xmax-xmin)
    else:
        xrng, yrng = None, None
        aratio = (ymax-ymin)/(xmax-xmin)

    plt = biggles.FramedPlot(
        xlabel='RA',
        ylabel='DEC',
        xrange=xrng,
        yrange=yrng,
        aspect_ratio=aratio,
    )

    allpts = biggles.Points(
        x, y,
        type=type,
    )
    plt.add(allpts)

    rev = hd['rev']
    icolor = 0
    for i in range(hd['hist'].size):
        if rev[i] != rev[i+1]:
            w = rev[rev[i]:rev[i+1]]
            if w.size >= minsize:
                indices = fof['number'][w]-1

                color = colors[icolor]
                xx = np.array(x[indices], ndmin=1)
                yy = np.array(y[indices], ndmin=1)

                pts = biggles.Points(
                    xx, yy,
                    type=fof_type,
                    size=fof_size,
                    color=color,
                )

                plt.add(pts)
                icolor += 1

    height = int(width*aratio)
    if plotfile is not None:
        ffront = os.path.basename(plotfile)
        name = ffront.split('-mof-')[0]
        plt.title = '%s FOF groups' % name

        print("writing:", plotfile)
        plt.write_img(width, int(height), plotfile)

    if show:
        plt.show(width=width, height=height)


def rainbow(num, type='hex'):
    """
    make rainbow colors

    parameters
    ----------
    num: integer
        number of colors
    type: string, optional
        'hex' or 'rgb', default hex
    """
    import colorsys

    def rgb_to_hex(rgb):
        return '#%02x%02x%02x' % rgb

    # not going to 360
    minh = 0.0
    # 270 would go to pure blue
    # maxh = 270.0
    maxh = 285.0

    if num == 1:
        hstep = 0
    else:
        hstep = (maxh-minh)/(num-1)

    colors = []
    for i in range(num):
        h = minh + i*hstep

        # just change the hue
        r, g, b = colorsys.hsv_to_rgb(h/360.0, 1.0, 1.0)
        r *= 255
        g *= 255
        b *= 255
        if type == 'rgb':
            colors.append((r, g, b))
        elif type == 'hex':

            rgb = (int(r), int(g), int(b))
            colors.append(rgb_to_hex(rgb))
        else:
            raise ValueError("color type should be 'rgb' or 'hex'")

    return colors
