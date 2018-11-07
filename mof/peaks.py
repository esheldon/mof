from numba import njit

@njit
def find_peaks(image, thresh, peakrows, peakcols):
    """
    find peaks by looking for points around which all values
    are lower

    image: 2d array
        An image in which to find peaks
    thresh: float
        Peaks must be higher than this value.  You would typically
        set this to some multiple of the noise level.
    peakrows: array
        an array to fill with peak locations
    peakcols: array
        an array to fill with peak locations
    """

    npeaks=0

    nrows, ncols = image.shape
    for irow in range(nrows):
        if irow==0 or irow==nrows-1:
            continue

        rowstart=irow-1
        rowend=irow+1
        for icol in range(ncols):
            if icol==0 or icol==ncols-1:
                continue

            colstart=icol-1
            colend=icol+1

            val = image[irow, icol]
            if val > thresh:

                ispeak=True
                for checkrow in range(rowstart,rowend+1):
                    for checkcol in range(colstart,colend+1):
                        if checkrow==irow and checkcol==icol:
                            continue

                        checkval = image[checkrow, checkcol]
                        if checkval > val:
                            # we found an adjacent value that is higher
                            ispeak=False

                            # break out of inner loop
                            break

                    if not ispeak:
                        # also break out of outer loop
                        break

                if ispeak:
                    npeaks += 1
                    ipeak = npeaks-1
                    peakrows[ipeak] = irow
                    peakcols[ipeak] = icol

    return npeaks
