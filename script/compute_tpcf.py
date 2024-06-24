#!/usr/bin/env python

import os
from argparse import ArgumentParser
import numpy as np
import astropy.io.fits as pyfits
from utils_shear_ana import catutil
import time
import schwimmbad
from constants import *
import treecorr
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_processor_count(pool, args):
    if isinstance(pool, schwimmbad.MPIPool):
        # MPIPool
        from mpi4py import MPI

        return MPI.COMM_WORLD.Get_size() - 1
    elif isinstance(pool, schwimmbad.MultiPool):
        # MultiPool
        return args.n_cores
    else:
        # SerialPool
        return 1


class TPCFWorker(object):
    def __init__(
        self,
        ntheta=17,
        min_sep=2.188,
        max_sep=332.954,
        window=False,
        overwrite=False,
    ):
        self.ntheta = ntheta
        self.min_sep = min_sep
        self.max_sep = max_sep
        self.cor = treecorr.GGCorrelation(
            nbins=ntheta,
            min_sep=self.min_sep,
            max_sep=self.max_sep,
            sep_units="arcmin",
            verbose=1,
        )
        self.rnom = self.cor.rnom
        self.window = window
        self.overwrite = overwrite

        self.output_folder = os.path.join("../data", f"output_2pcf_{self.ntheta}")
        os.makedirs(self.output_folder, exist_ok=True)

        self.patch_file = os.path.join("../data", "hscy3_patches_xxx.fits")
        for i in range(NZ_BINS):
            patch_file = self.patch_file.replace("xxx", "%d" % i)
            if not os.path.exists(patch_file):
                data = self.load_data(i)
                self.make_patches(data, patch_file)

    def run(self, idx):
        start = time.time()
        i, j = BIN_PAIRS[idx]
        print(f"Running TPCF with {self.ntheta} bins for bin pair ({i}, {j})")

        tpcf = np.zeros((7, self.ntheta), dtype=float)
        tpcf_file = os.path.join(self.output_folder, f"tpcf_{i}_{j}.fits")
        if os.path.exists(tpcf_file) and not self.overwrite:
            print(f"{tpcf_file} exists and overwrite is set to False")
            return

        gg = treecorr.GGCorrelation(
            nbins=ntheta,
            min_sep=self.min_sep,
            max_sep=self.max_sep,
            sep_units="arcmin",
            verbose=1,
        )

        data_i = self.load_data(i)
        patch_file_i = self.patch_file.replace("xxx", "%d" % i)
        cat_i = self.convert_data2treecat(data_i, patch_file_i)
        N_i = len(data_i["g1"])
        cat_i.get_patches()
        # Auto power spectra
        if i == j:
            gg.process(cat_i, comm=comm)
            N_j = N_i
        # Cross power spectra
        else:
            data_j = self.load_data(j)
            patch_file_j = self.patch_file.replace("xxx", "%d" % j)
            cat_j = self.convert_data2treecat(data_j, patch_file_j)
            N_j = len(data_j["g1"])
            gg.process(cat_i, cat_j, comm=comm)

        comm.Barrier()

        if rank == 0:
            tpcf[0] = gg.meanr
            tpcf[1] = gg.xip
            tpcf[2] = gg.xim
            tpcf[3] = gg.varxip
            tpcf[4] = gg.varxim
            tpcf[5] = gg.npairs
            tpcf[6] = gg.weight

            pyfits.writeto(tpcf_file, tpcf, overwrite=True)

            if self.window:
                xi_W = self.run_xi_W(gg.meanr, gg.bin_size, gg.weight, N_i, N_j)
                tpcfw_file = os.path.join(self.output_folder, f"tpcfw_{i}_{j}.fits")
                if os.path.exists(tpcfw_file) and not self.overwrite:
                    print(f"Window {tpcfw_file} exists and overwrite is set to False")
                    return
                pyfits.writeto(tpcfw_file, xi_W, overwrite=True)

        end = time.time()
        print(
            f"ntheta={self.ntheta} for bin pair ({i}, {j}) took {end - start} seconds"
        )
        return

    def run_xi_W(self, theta, theta_bin_size, weight, N_i, N_j):
        theta_rad = np.pad(
            theta * ARCMIN2RAD,
            (1, 1),
            "constant",
            constant_values=(self.min_sep * ARCMIN2RAD, self.max_sep * ARCMIN2RAD),
        )
        bin_area = 2.0 * np.pi * (np.cos(theta_rad[:-1]) - np.cos(theta_rad[1:]))
        expected_weight = bin_area[:-1] * N_i * N_j * FSKY
        xi_W = weight / expected_weight
        xi_W /= xi_W[0]
        return xi_W

    def load_data(self, z):
        data = pyfits.getdata(f"../data/data_z{z+1}.fits")
        wsum = np.sum(data["i_hsmshaperegauss_derived_weight"])
        mbias = (
            np.sum(
                data["i_hsmshaperegauss_derived_shear_bias_m"]
                * data["i_hsmshaperegauss_derived_weight"]
            )
            / wsum
        )
        msel, asel, msel_err, asel_err = catutil.get_sel_bias(
            data["i_hsmshaperegauss_derived_weight"],
            data["i_apertureflux_10_mag"],
            data["i_hsmshaperegauss_resolution"],
        )
        g1, g2 = catutil.get_shear_regauss(data, mbias, msel, asel)
        ra, dec = catutil.get_radec(data)
        e1, e2 = catutil.get_gal_ellip(data)
        return {
            "g1": g1,
            "g2": g2,
            "ra": ra,
            "dec": dec,
            "e1": e1,
            "e2": e2,
            "e_rms": data["i_hsmshaperegauss_derived_rms_e"],
            "sigma_e": data["i_hsmshaperegauss_derived_sigma_e"],
            "weight": data["i_hsmshaperegauss_derived_weight"],
        }

    def convert_data2treecat(self, data, patch_file):
        cat = treecorr.Catalog(
            g1=data["g1"],
            g2=-data["g2"],
            ra=data["ra"],
            dec=data["dec"],
            w=data["weight"],
            ra_units="deg",
            dec_units="deg",
            patch_centers=patch_file,
            verbose=1,
        )
        return cat

    def make_patches(self, data, patch_file):
        if not os.path.exists(self.patch_file):
            print("Making patches")
            cat = treecorr.Catalog(
                g1=data["g1"],
                g2=-data["g2"],
                ra=data["ra"],
                dec=data["dec"],
                w=data["weight"],
                ra_units="deg",
                dec_units="deg",
                npatch=28,
                verbose=2,
            )
            cat.get_patches()
            cat.write_patch_centers(patch_file)
        else:
            print("Using existing patch file")


if __name__ == "__main__":
    parser = ArgumentParser(description="simulate blended images")
    parser.add_argument(
        "--ntheta", type=int, default=17, help="number of bins to use for treecorr"
    )
    parser.add_argument(
        "--min_sep", type=float, default=2.188, help="minimum sep angle in arcmin"
    )
    parser.add_argument(
        "--max_sep", type=float, default=332.954, help="maximum sep angle in arcmin"
    )
    parser.add_argument(
        "--window", default=False, type=bool, help="whether to compute xi_W"
    )
    parser.add_argument(
        "--idx", type=int, default=0, help="idx of bin pair to compute 2pcf"
    )

    args = parser.parse_args()
    ntheta = args.ntheta
    min_sep = args.min_sep
    max_sep = args.max_sep
    window = args.window
    idx = args.idx

    start = time.time()

    worker = TPCFWorker(ntheta, min_sep, max_sep, window)
    worker.run(idx)

    end = time.time()

    print(f"TPCF took {end - start} seconds")
