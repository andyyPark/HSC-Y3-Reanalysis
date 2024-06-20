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
        overwrite=False,
    ):
        self.ntheta = ntheta
        self.min_sep = min_sep
        self.max_sep = max_sep
        self.cor = treecorr.GGCorrelation(
            nbins=ntheta, min_sep=self.min_sep, max_sep=self.max_sep, sep_units="arcmin"
        )
        self.rnom = self.cor.rnom
        self.overwrite = overwrite

        self.output_folder = os.path.join("../data", f"output_2pcf_{self.ntheta}")
        os.makedirs(self.output_folder, exists=True)

    def run(self, idx):
        start = time.time()
        i, j = BIN_PAIRS[idx]
        print(f"Running TPCF with {self.ntheta} bins for bin pair ({i}, {j})")

        tpcf = np.zeros((6, self.ntheta), dtype=float)
        tpcf_file = os.path.join(self.output_folder, f"tpcf_{i}_{j}.fits")
        if os.path.exists(tpcf_file) and not self.overwrite:
            print(f"{tpcf_file} exists and overwrite is set to False")
            return

        data_i = self.load_data(i)
        cat_i = self.convert_data2treecat(data_i)
        # Auto power spectra
        if i == j:
            self.cor.clear()
            self.cor.process(cat_i, cat_i)
        # Cross power spectra
        else:
            data_j = self.load_data(j)
            cat_j = self.convert_data2treecat(data_j)
            self.cor.clear()
            self.cor.process(cat_i, cat_j)

        tpcf[0] = self.cor.xip
        tpcf[1] = self.cor.xim
        tpcf[2] = self.cor.varxip
        tpcf[3] = self.cor.varxim
        tpcf[4] = self.cor.npairs
        tpcf[5] = self.cor.weight

        pyfits.writeto(tpcf_file, tpcf, overwrite=True)

        end = time.time()
        print(
            f"ntheta={self.ntheta} for bin pair ({i}, {j}) took {end - start} seconds"
        )
        return

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

    def convert_data2treecat(self, data):
        tree_cat = treecorr.Catalog(
            g1=data["g1"],
            g2=-data["g2"],
            ra=data["ra"],
            dec=data["dec"],
            w=data["weight"],
            ra_units="deg",
            dec_units="deg",
        )
        return tree_cat


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
        "--auto", default=False, type=bool, help="whether to only compute auto"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--ncores",
        dest="n_cores",
        default=1,
        type=int,
        help="Number of processes (uses multiprocessing).",
    )
    group.add_argument(
        "--mpi",
        dest="mpi",
        default=False,
        action="store_true",
        help="Run with MPI.",
    )

    args = parser.parse_args()
    ntheta = args.ntheta
    min_sep = args.min_sep
    max_sep = args.max_sep
    auto = args.auto

    idx = np.array([0, 4, 7, 9]) if auto else np.arange(10)

    start = time.time()

    worker = TPCFWorker(ntheta, min_sep, max_sep)
    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    ncores = get_processor_count(pool, args)
    print(f"Running with mpi={args.mpi} with ncores = {ncores}")
    print(f"ntheta = {ntheta}, min_sep = {min_sep}, max_sep = {max_sep} auto = {auto}")

    pool.map(worker.run, idx)

    end = time.time()
    pool.close()

    print(f"TPCF took {end - start} seconds")
