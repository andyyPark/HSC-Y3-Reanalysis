#!/usr/bin/env python

import os
from argparse import ArgumentParser
import numpy as np
import healpy as hp
import astropy.io.fits as pyfits
from utils_shear_ana import catutil
import pymaster as nmt
import time
import schwimmbad

NZ_BINS = 4
NZ_PAIRS = int(NZ_BINS * (NZ_BINS + 1) / 2)
BIN_PAIRS = [(i, j) for i in range(NZ_BINS) for j in range(i, NZ_BINS)]
BP2INDEX = {bp: i for i, bp in enumerate(BIN_PAIRS)}
STR2DEG = 4 * np.pi * (180 / np.pi) ** 2
FSKY = 416 / STR2DEG


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

class PowerWorker(object):

    def __init__(self, nside, overwrite=False):
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.overwrite = overwrite
        self.ell_max = 3 * nside

    def run(self, idx):
        start = time.time()
        i, j = BIN_PAIRS[idx]
        print(f"Running nside={nside} for bin pair ({i}, {j})")
        power = np.zeros((3, self.ell_max), dtype=float)

        # Auto power spectra
        if i == j:
            data_i = self.load_data(i)
            pixels_i = hp.ang2pix(self.nside, data_i["ra"], data_i["dec"], lonlat=True)
            mask_i, g1_map_i, g2_map_i = self.get_map(pixels_i, data_i["g1"], data_i["g2"], False, i)
            field_i = nmt.NmtField(mask_i, [g1_map_i, g2_map_i], spin=2, lite=True)
            cl_coupled = nmt.compute_coupled_cell(field_i, field_i)
            power[0] = cl_coupled[0] # EE
            power[1] = cl_coupled[3] # BB
            power[2] = cl_coupled[1] # EB

        power_file = os.path.join("../data", f"pseudo_ps_{nside}_{i}_{j}.fits")
        pyfits.writeto(power_file, power, overwrite=self.overwrite)

        end = time.time()

        print(f"nside={nside} for bin pair ({i}, {j}) took {end - start} seconds")
        return

    def load_data(self, z):
        data = pyfits.getdata(f"../data/data_z{z+1}.fits")
        wsum = np.sum(data['i_hsmshaperegauss_derived_weight'])
        mbias = np.sum(
            data['i_hsmshaperegauss_derived_shear_bias_m']
            * data['i_hsmshaperegauss_derived_weight']
        ) / wsum
        msel, asel, msel_err, asel_err = catutil.get_sel_bias(
            data['i_hsmshaperegauss_derived_weight'],
            data['i_apertureflux_10_mag'],
            data['i_hsmshaperegauss_resolution'],
        )
        g1, g2 = catutil.get_shear_regauss(data, mbias, msel, asel)
        ra, dec = catutil.get_radec(data)
        e1, e2 = catutil.get_gal_ellip(data)
        return {
                "g1": g1, "g2": g2,
                "ra": ra, "dec": dec,
                "e1": e1, "e2": e2,
                "e_rms": data["i_hsmshaperegauss_derived_rms_e"],
                "sigma_e": data["i_hsmshaperegauss_derived_sigma_e"]
                }

    def get_map(self, pixels, g1, g2, save_window, i):
        start = time.time()
        print(f"Getting map {i}")
        Ngal = np.bincount(pixels, minlength=self.npix)
        Nbar = np.average(Ngal[Ngal != 0])
        g1_map = np.bincount(pixels, weights=g1, minlength=self.npix) / Nbar
        g2_map = np.bincount(pixels, weights=g2, minlength=self.npix) / Nbar
        g1_map[Ngal == 0] = hp.UNSEEN
        g2_map[Ngal == 0] = hp.UNSEEN
        mask = np.zeros(self.npix, dtype=int)
        mask[pixels] = 1

        if save_window:
            window = Ngal / Nbar
            window[Ngal == 0] = hp.UNSEEN
            window_file = os.path.join("../data", f"window_{i}_{self.nside}.fits")
            if os.path.exists(window_file):
                print(f"Window map for redshift {i} exists")
            else:
                pyfits.writeto(window_file, window)
        end = time.time()
        print(f"Finished getting map {i} in {end - start} seconds.")

        return mask, g1_map, g2_map


def main_old(nside, power, save_window):
    npix = hp.nside2npix(nside)
    power = np.zeros((NZ_PAIRS, 3, 3*nside), dtype=float)
    noi_power = np.zeros((NZ_BINS, 2, 3*nside))
    for i in range(NZ_BINS):
        data_i = load_data(i)
        pixels_i = hp.ang2pix(nside, data_i["ra"], data_i["dec"], lonlat=True)
        mask_i, g1_map_i, g2_map_i = self.get_map(npix, pixels_i, data_i["g1"], data_i["g2"], save_window, i)
        field_i = nmt.NmtField(mask_i, [g1_map_i, g2_map_i], spin=2, lite=True)
        for j in range(i, NZ_BINS):
            bp = (i, j)
            data_j = load_data(j)
            pixels_j = hp.ang2pix(nside, data_j["ra"], data_j["dec"], lonlat=True)
            mask_j, g1_map_j, g2_map_j = get_map(npix, pixels_j, data_j["g1"], data_j["g2"], save_window, j)
            field_j = nmt.NmtField(mask_j, [g1_map_j, g2_map_j], spin=2, lite=True)
            cl_coupled = nmt.compute_coupled_cell(field_i, field_j)
            power[BP2INDEX[bp], 0] = cl_coupled[0] ## EE
            power[BP2INDEX[bp], 1] = cl_coupled[3] ## BB
            power[BP2INDEX[bp], 2] = cl_coupled[1] ## EB

            if i == j:
                noise_ps = compute_noi(data_i, nside, i)
                noi_power[i, 0] = noise_ps[0]
                noi_power[i, 1] = noise_ps[1]

    power_file = os.path.join("../data", f"pseudo_ps_{nside}.fits")
    pyfits.writeto(power_file, power)

    noise_file = os.path.join("../data", f"noise_ps_{i}_{nside}.fits")
    pyfits.writeto(noise_file, noi_power)

def compute_noi(data, nside, i, N=100):
    noi_power = np.zeros((2, N, 3*nside))
    npix = hp.nside2npix(nside)
    pixels = hp.ang2pix(nside, data["ra"], data["dec"], lonlat=True)
    print(data)
    for seed in range(N):
        int1, int2, meas1, meas2 = catutil.simulate_shape_noise(
            e1=data["e1"],
            e2=data["e2"],
            e_rms=data["e_rms"],
            sigma_e=data["sigma_e"],
            seed=seed,
        )
        zero_shear_sig = np.zeros_like(int1)
        e1_mock, e2_mock = catutil.generate_mock_shape_from_sim(
            gamma1_sim=zero_shear_sig,
            gamma2_sim=zero_shear_sig,
            kappa_sim=zero_shear_sig,
            shape1_int=int1,
            shape2_int=int2,
            shape1_meas=meas1,
            shape2_meas=meas2,
        )
        mask, g1_map, g2_map = self.get_map(npix, pixels, e1_mock, e2_mock, save_window=False, i=None)
        field = nmt.NmtField(mask, [g1_map, g2_map], spin=2, lite=True)
        cl_coupled = nmt.compute_coupled_cell(field, field)
        noi_power[0, seed] = cl_coupled[0] ## EE
        noi_power[1, seed] = cl_coupled[3] ## BB
    noi_power = np.average(noi_power, axis=1)
    return noi_power



if __name__ == "__main__":
    parser = ArgumentParser(description="simulate blended images")
    parser.add_argument(
            "--nside",
            type=int,
            default=128,
            help="Nside to be used for generating maps"
    )
    parser.add_argument(
        "--power",
        default=True,
        type=bool,
        help="whether to save the power spectra (pseudo)",
    )
    parser.add_argument(
        "--noise",
        default=True,
        type=bool,
        help="whether to compute the noise power spectra (pseudo)",
    )
    parser.add_argument(
        "--save_window",
        default=True,
        type=bool,
        help="whether to save the window function",
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
    nside = args.nside
    power = args.power
    save_window = args.save_window
    
    worker = PowerWorker(nside, False)
    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    ncores = get_processor_count(pool, args)
    print(f"Running with mpi={args.mpi} with ncores = {ncores}")
    pool.map(worker.run, np.array([0, 4, 7, 9]))
    pool.close()

