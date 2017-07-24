import numpy as np, healpy as hp, matplotlib.pyplot as plt
import os
import argparse
import re

def healpixellize(f_in, theta_in, phi_in, nside):
    f = f_in.flatten()
    theta = theta_in.flatten()
    phi = phi_in.flatten()

    pix = hp.ang2pix(nside,theta,phi)

    hmap = np.zeros(hp.nside2npix(nside))
    hits = np.zeros(hp.nside2npix(nside))


    for i,v in enumerate(f):
        hmap[pix[i]] += v
        hits[pix[i]] +=1
    hmap = hmap/hits

    return hmap

def AzimuthalRotation(hmap):
    """
    Azimuthal rotation of a healpix map by pi/2 about the z-axis
    """
    npix = len(hmap)
    nside= hp.npix2nside(npix)
    hpxidx = np.arange(npix)
    t2,p2 = hp.pix2ang(nside, hpxidx)

    p = p2 - np.pi/2
    p[p < 0] += 2. * np.pi
    t = t2

    idx = hp.ang2pix(nside, t, p)

    hout = hmap[idx]
    return hout

def StokesMatrix(n):
    if n not in [0,1,2,3]: raise Exception('Input must be an integer in [0,1,2,3]')

    if n == 0:
        p = np.array([[1.,0],[0.,1.]])
    elif n == 1:
        p = np.array([[1.,0],[0,-1.]])
    elif n == 2:
        p = np.array([[0,1.],[1.,0]])
    elif n == 3:
        p = np.array([[0., -1j],[1j,0]])

    return p

def MuellerMatrixElement(J,i,j):

    Pi = StokesMatrix(i)
    Pj = StokesMatrix(j)

    M_ij = (1./2.) * np.einsum('...ab,...bc,...cd,...ad',Pi,J,Pj,J.conj())

    M_ij = np.real(M_ij)

    return M_ij

class LeakageMetric(object):
    """
    Inputs:

    filenames: A list of strings, each of which is the full path to a CST
        output textfile. Each filename must contain a substring "f=" or "MHz"
        to determine the frequency of the file, formatted as in these two
        examples:
            ".../HERA_sim_120MHz.txt"
            ".../farfield (f=68) [1].txt"

    normalization: a string, choices are "peak", "integral", or "none".
        -"peak" normalizes the Mueller matrix by the maximum of I->I for each
         frequency.
        -"integral" normalizes the Mueller matrix by the integral of I->I for
         each frequency.
        -"none" does no normalization, so the Mueller matrix units which are
         the square of the units U of the input E-field data. Then the leakage
         integral has units of U^4.

    The result is a function of frequency stored in the attribute leakage_bound.
    The frequency axis is stored in freqs.
    """
    def __init__(self, filenames, normalization, filename2freq=None):
        if normalization not in ['peak', 'integral', 'none']:
            raise ValueError('normalization must be "peak", "integral", or "none" ')

        if not all([os.path.exists(f) for f in filenames]):
            raise ValueError('At least one input filename is not valid.')

        if any(['MHz' not in f for f in filenames]) and any(['f=' not in f for f in filenames]):
            raise ValueError('Cannot determine the frequency from the filename of a least one input file')

        if not isinstance(filenames, (list, tuple)):
            filenames = [filenames]

        self.filenames = filenames

        self.freqs = []
        if all(['MHz' in fname for fname in filenames]):
            for fname in filenames:
                f = re.findall('\d+' + 'MHz', fname)[0][:-3]
                self.freqs.append(float(f))
        elif all(['f=' in fname for fname in filenames]):
            for fname in filenames:
                f = re.findall('f=' + '\d+', fname)[0][2:]
                self.freqs.append(float(f))

        self.filenames = [x for (y,x) in sorted(zip(self.freqs, self.filenames), key=lambda t: t[0])]
        self.freqs.sort()

        L = []
        Norms = []
        for fname in self.filenames:
            J = self.make_jones(fname)
            L_nu, norm = self.integrate_leakage(J)
            L_nu /= norm[normalization]**2.
            L.append(L_nu)
            Norms.append(norm[normalization])

        self.leakage_bound = np.array(L)
        self.freqs = np.array(self.freqs)

    def make_jones(self, filename):
        data = np.loadtxt(filename, skiprows=2)
        theta_data = np.radians(data[:,0])
        phi_data = np.radians(data[:,1])

        Et = data[:,3] * np.exp(-1j * np.radians(data[:,4]))
        Ep = data[:,5] * np.exp(-1j * np.radians(data[:,6]))

        cosp = np.cos(phi_data)
        sinp = np.sin(phi_data)

        rEt = cosp * Et - sinp * Ep
        rEp = sinp * Et + cosp * Ep

        theta_f,phi_f = np.abs(theta_data), np.where(theta_data < 0, phi_data + np.pi, phi_data)

        nside = 32
        npix = hp.nside2npix(nside)
        hpxidx = np.arange(npix)
        theta, phi = hp.pix2ang(nside, hpxidx)
        phi = np.where(phi >= np.pi, phi - np.amax(phi), phi)

        hpxiz = lambda m: healpixellize(m,theta_f,phi_f,nside)

        EXt, EXp = [hpxiz(X.real) + 1j * hpxiz(X.imag) for X in [rEt, rEp]]

        cosP = np.cos(phi)
        sinP = np.sin(phi)

        nEXt = cosP * EXt + sinP * EXp
        nEXp = -sinP * EXt + cosP * EXp

        nEYt = AzimuthalRotation(nEXt)
        nEYp = AzimuthalRotation(nEXp)

        jones_out = np.array([[nEXt,nEXp],[nEYt,nEYp]]).transpose(2,0,1)

        return jones_out

    def integrate_leakage(self, jones):
        npix = jones.shape[0]
        nside = hp.npix2nside(npix)
        hpxidx = np.arange(npix)
        theta, phi = hp.pix2ang(nside, hpxidx)

        M00, M01, M02, M03 = [MuellerMatrixElement(jones, 0, k) for k in range(4)]

        hm = np.zeros(npix)
        hm[np.where(theta <= np.pi/2.)] = 1

        norms = {}
        norms['none'] = 1.
        norms['integral'] = (4. * np.pi/npix) * np.sum(hm * M00)
        norms['peak'] = np.amax(M00)

        L = 0
        for M in [M01, M02, M03]:
            L += (4. * np.pi/npix) * np.sum(hm * (M)**2.)

        return L, norms

    def compute_solid_angle(self):
        S = []
        for fname in self.filenames:
            J = self.make_jones(fname)
            S.append(self.M00_solid_angle_integral(J))
        self.solid_angle_spectrum = np.array(S)

    def M00_solid_angle_integral(self, jones):
        npix = jones.shape[0]
        nside = hp.npix2nside(npix)
        hpxidx = np.arange(npix)
        theta, phi = hp.pix2ang(nside, hpxidx)

        M00 = MuellerMatrixElement(jones, 0, 0)

        hm = np.zeros(npix)
        hm[np.where(theta <= np.pi/2.)] = 1

        integral = (4. * np.pi/npix) * np.sum(hm * M00) / np.amax(M00)
        return integral

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('-N', '--normalization', required=True,
        help='The beam normalization to use. "peak" divides by the maximum at\
            each frequency. "integral" divides by the solid angle per frequency.\
            "none" performs no normalization so that the units of the computed \
            leakage metric will be the square of the input E-field units.',
        dest='normalization')

    p.add_argument('-f', '--filenames', dest='filenames', nargs='*', required=True,
        help='The files to read, one per frequency.')

    p.add_argument('-o', '--outfile', dest='outfile',
        help='Output filename')

    p.add_argument('--plot-lin',action='store_true', dest='make_lin_plot',
        help='Plots the result, linear scale')

    p.add_argument('--plot-log',action='store_true', dest='make_log_plot',
        help='Plots the result, log scale')

    args = p.parse_args()
    LM = LeakageMetric(args.filenames, args.normalization)

    if args.outfile is not None:
        np.savetxt(args.outfile, np.c_[LM.freqs, LM.leakage_bound])
    else:
        np.savetxt('leakage_metric_output.txt', np.c_[LM.freqs, LM.leakage_bound])

    if args.make_log_plot or args.make_lin_plot is True:
        lin_func = lambda x: x
        type_func = np.log10 if args.make_log_plot is True else lin_func
        plt.figure()
        plt.plot(LM.freqs, type_func(LM.leakage_bound), c='k')
        plt.xlabel('Frequency (MHz)')
        plt.title('Leakage bound, {} normalization'.format(args.normalization))
        plt.show()
