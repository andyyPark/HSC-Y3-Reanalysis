import os
import logging
import numpy as np
import healpy as hp


class ShearMap(object):
    def __init__(self, nside, cat):
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.cat = cat
        self.ra = cat["ra"]
        self.dec = cat["dec"]
        self.g1 = cat["g1"]
        self.g2 = cat["g2"]
        self.e1 = cat["e1"]
        self.e2 = cat["e2"]
        self.weight = cat["weight"]
        self.pixels = hp.ang2pix(nside, cat["ra"], cat["dec"], lonlat=True)
        self.count_map = np.bincount(self.pixels, minlength=self.npix)
        self.Nbar = np.average(self.count_map[self.count_map != 0])

    def SN(
        self,
    ):
        w = self.cat["weight"]
        erms = self.cat["e_rms"]
        sigma_e = self.cat["sigma_e"]
        wsum = np.sum(w)
        res = 1.0 - np.sum(erms**2.0 * w) / wsum
        num = np.sum(w * (erms**2 + sigma_e**2))
        denom = wsum  # * (2.0 * res) ** 2
        return num / denom

    def run(self, cnorm=True):
        g1_weights = self.g1 * self.weight
        g2_weights = self.g2 * self.weight
        g1_map = np.bincount(self.pixels, weights=g1_weights, minlength=self.npix)
        g2_map = np.bincount(self.pixels, weights=g2_weights, minlength=self.npix)
        weight_map = np.bincount(self.pixels, weights=self.weight, minlength=self.npix)
        if cnorm:
            g1_map /= self.Nbar
            g2_map /= self.Nbar
        else:
            g1_map /= weight_map
            g2_map /= weight_map
        g1_map[np.isnan(g1_map)] = hp.UNSEEN
        g2_map[np.isnan(g2_map)] = hp.UNSEEN
        w = np.where(g2_map != hp.UNSEEN)
        g2_map[w] *= -1
        return g1_map, g2_map

    def run_noise(self, g=True):
        if g:
            noise_weight = self.weight**2 * 0.5 * (self.g1**2 + self.g2**2)
        else:
            noise_weight = self.weight**2 * 0.5 * (self.e1**2 + self.e2**2)
        noise_map = np.bincount(self.pixels, weights=noise_weight, minlength=self.npix)
        noise_map[np.isnan(noise_map)] = hp.UNSEEN
        return noise_map

    def window(self, cnorm=True):
        if cnorm:
            SW = self.count_map / self.Nbar
        else:
            weight_map = np.bincount(
                self.pixels, weights=self.weight, minlength=self.npix
            )
            SW = self.count_map / weight_map
            SW[np.isnan(SW)] = hp.UNSEEN
        return SW

    def mask(self):
        weight_map = np.bincount(self.pixels, weights=self.weight, minlength=self.npix)
        mask = np.zeros(self.npix, dtype=bool)
        mask |= weight_map > 0
        return mask
