import numpy as np


class PriorSimpleSepMulti(object):
    """
    different center priors for each object, same priors
    for the structural and flux parameters
    """
    def __init__(self,
                 cen_priors,
                 g_prior,
                 T_prior,
                 F_prior):

        self.nobj = len(cen_priors)
        self.cen_priors = cen_priors
        self.g_prior = g_prior
        self.T_prior = T_prior

        if isinstance(F_prior, list):
            self.nband = len(F_prior)
        else:
            self.nband = 1
            F_prior = [F_prior]

        self.npars_per = 5+self.nband
        self.F_priors = F_prior

    def fill_fdiff(self, allpars, fdiff, **keys):
        """
        set sqrt(-2ln(p)) ~ (model-data)/err
        """
        index = 0

        fstart = 0
        for i in range(self.nobj):

            fstart = index

            beg = i*self.npars_per
            end = (i+1)*self.npars_per

            pars = allpars[beg:end]

            cen_prior = self.cen_priors[i]

            lnp1, lnp2 = cen_prior.get_lnprob_scalar_sep(pars[0], pars[1])

            fdiff[index] = lnp1
            index += 1
            fdiff[index] = lnp2
            index += 1

            fdiff[index] = self.g_prior.get_lnprob_scalar2d(pars[2], pars[3])
            index += 1

            fdiff[index] = self.T_prior.get_lnprob_scalar(pars[4], **keys)
            index += 1

            for j in range(self.nband):
                F_prior = self.F_priors[j]
                fdiff[index] = F_prior.get_lnprob_scalar(pars[5+j])
                index += 1

            chi2 = -2*fdiff[fstart:index].copy()
            chi2.clip(min=0.0, max=None, out=chi2)
            fdiff[fstart:index] = np.sqrt(chi2)

        return index

    def get_prob_scalar(self, pars, **keys):
        """
        probability for scalar input (meaning one point)
        """

        lnp = self.get_lnprob_scalar(pars, **keys)
        p = np.exp(lnp)
        return p

    def get_lnprob_scalar(self, allpars, **keys):
        """
        log probability for scalar input (meaning one point)
        """

        lnp = 0.0
        for i in range(self.nobj):

            beg = i*self.npars_per
            end = (i+1)*self.npars_per

            pars = allpars[beg:end]

            cen_prior = self.cen_priors[i]
            lnp += cen_prior.get_lnprob_scalar(pars[0], pars[1])
            lnp += self.g_prior.get_lnprob_scalar2d(pars[2], pars[3])
            lnp += self.T_prior.get_lnprob_scalar(pars[4], **keys)

            for j, F_prior in enumerate(self.F_priors):
                lnp += F_prior.get_lnprob_scalar(pars[5+j], **keys)

        return lnp

    def sample(self):
        """
        Get random samples
        """

        samples = np.zeros(self.npars_per*self.nobj)

        for i in range(self.nobj):
            cen_prior = self.cen_priors[i]

            cen1, cen2 = cen_prior.sample()
            g1, g2 = self.g_prior.sample2d()
            T = self.T_prior.sample()

            beg = i*self.npars_per

            samples[beg+0] = cen1
            samples[beg+1] = cen2
            samples[beg+2] = g1
            samples[beg+3] = g2
            samples[beg+4] = T

            for j, F_prior in enumerate(self.F_priors):
                F = F_prior.sample()
                samples[beg+5+j] = F

        return samples


class PriorBDFSepMulti(object):
    """
    different center priors for each object, same priors
    for the structural and flux parameters
    """
    def __init__(self,
                 cen_priors,
                 g_prior,
                 T_prior,
                 fracdev_prior,
                 F_prior):

        self.nobj = len(cen_priors)
        self.cen_priors = cen_priors
        self.g_prior = g_prior
        self.T_prior = T_prior
        self.fracdev_prior = fracdev_prior

        if isinstance(F_prior, list):
            self.nband = len(F_prior)
        else:
            self.nband = 1
            F_prior = [F_prior]

        self.npars_per = 6+self.nband
        self.F_priors = F_prior

        self.set_bounds()

    def set_bounds(self):
        """
        set possibe bounds
        """
        bounds = [
            (None, None),  # c1
            (None, None),  # c2
            (None, None),  # g1
            (None, None),  # g2
        ]

        allp = [
            self.T_prior,
            self.fracdev_prior,
        ] + self.F_priors

        some_have_bounds = False
        for i, p in enumerate(allp):
            if p.has_bounds():
                some_have_bounds = True
                bounds.append((p.bounds[0], p.bounds[1]))
            else:
                bounds.append((None, None))

        if not some_have_bounds:
            bounds = None

        self.bounds = bounds

    def fill_fdiff(self, allpars, fdiff, **keys):
        """
        set sqrt(-2ln(p)) ~ (model-data)/err
        """
        index = 0

        fstart = 0
        for i in range(self.nobj):

            fstart = index

            beg = i*self.npars_per
            end = (i+1)*self.npars_per

            pars = allpars[beg:end]

            cen_prior = self.cen_priors[i]
            lnp1, lnp2 = cen_prior.get_lnprob_scalar_sep(pars[0], pars[1])

            fdiff[index] = lnp1
            index += 1
            fdiff[index] = lnp2
            index += 1

            fdiff[index] = self.g_prior.get_lnprob_scalar2d(pars[2], pars[3])
            index += 1
            fdiff[index] = self.T_prior.get_lnprob_scalar(pars[4], **keys)
            index += 1

            fdiff[index] = self.fracdev_prior.get_lnprob_scalar(
                pars[5],
                **keys
            )
            index += 1

            for j in range(self.nband):
                F_prior = self.F_priors[j]
                fdiff[index] = F_prior.get_lnprob_scalar(pars[6+j], **keys)
                index += 1

            chi2 = -2*fdiff[fstart:index].copy()
            chi2.clip(min=0.0, max=None, out=chi2)
            fdiff[fstart:index] = np.sqrt(chi2)

        return index

    def get_prob_scalar(self, pars, **keys):
        """
        probability for scalar input (meaning one point)
        """

        lnp = self.get_lnprob_scalar(pars, **keys)
        p = np.exp(lnp)
        return p

    def get_lnprob_scalar(self, allpars, **keys):
        """
        log probability for scalar input (meaning one point)
        """

        lnp = 0.0
        for i in range(self.nobj):

            beg = i*self.npars_per
            end = (i+1)*self.npars_per

            pars = allpars[beg:end]

            cen_prior = self.cen_priors[i]
            lnp += cen_prior.get_lnprob_scalar(pars[0], pars[1])
            lnp += self.g_prior.get_lnprob_scalar2d(pars[2], pars[3])
            lnp += self.T_prior.get_lnprob_scalar(pars[4], **keys)
            lnp += self.fracdev_prior.get_lnprob_scalar(pars[5], **keys)

            for j, F_prior in enumerate(self.F_priors):
                lnp += F_prior.get_lnprob_scalar(pars[6+j], **keys)

        return lnp

    def sample(self):
        """
        Get random samples
        """

        samples = np.zeros(self.npars_per*self.nobj)

        for i in range(self.nobj):
            cen_prior = self.cen_priors[i]

            cen1, cen2 = cen_prior.sample()
            g1, g2 = self.g_prior.sample2d()
            T = self.T_prior.sample()
            fracdev = self.fracdev_prior.sample()

            beg = i*self.npars_per

            samples[beg+0] = cen1
            samples[beg+1] = cen2
            samples[beg+2] = g1
            samples[beg+3] = g2
            samples[beg+4] = T
            samples[beg+5] = fracdev

            for j, F_prior in enumerate(self.F_priors):
                F_prior = self.F_priors[j]
                F = F_prior.sample()
                samples[beg+6+j] = F

        return samples
