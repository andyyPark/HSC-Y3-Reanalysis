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
from constants import *
from maps import ShearMap


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
    def __init__(
        self,
        nside,
        power=True,
        noise=False,
        noise_analytic=False,
        save_window=False,
        overwrite=False,
    ):
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.power = power
        self.noise = noise
        self.noise_analytic = noise_analytic
        self.save_window = save_window
        self.overwrite = overwrite
        self.ell_max = 3 * nside
        self.Omega_pix = hp.nside2pixarea(nside, degrees=False)

        self.output_folder = os.path.join("../data", f"output_{nside}")
        os.makedirs(self.output_folder, exist_ok=True)

    def run(self, idx):
        if not self.power:
            return
        start = time.time()
        i, j = BIN_PAIRS[idx]
        print(f"Running nside={self.nside} for bin pair ({i}, {j})")
        power = np.zeros((3, self.ell_max), dtype=float)
        power_file = os.path.join(self.output_folder, f"pseudo_ps_{i}_{j}.fits")
        if os.path.exists(power_file) and not self.overwrite:
            print(f"{power_file} exists and overwrite is set to False")
            return

        data_i = self.load_data(i)
        sm_i = ShearMap(self.nside, data_i)
        window_file = os.path.join(self.output_folder, f"window_{i}.fits")
        if self.save_window and not os.path.exists(window_file):
            window = sm_i.window()
            pyfits.writeto(window_file, window)

        g1_map_i, g2_map_i = sm_i.run()
        mask_i = sm_i.mask()
        field_i = nmt.NmtField(mask_i, [g1_map_i, g2_map_i], spin=2, lite=True)

        # Auto power spectra
        if i == j:
            cl_coupled = nmt.compute_coupled_cell(field_i, field_i)
        # Cross power spectra
        else:
            data_j = self.load_data(j)
            sm_j = ShearMap(self.nside, data_j)
            g1_map_j, g2_map_j = sm_j.run()
            mask_j = sm_j.mask()
            field_j = nmt.NmtField(mask_j, [g1_map_j, g2_map_j], spin=2, lite=True)
            cl_coupled = nmt.compute_coupled_cell(field_i, field_j)

        power[0] = cl_coupled[0]  # EE
        power[1] = cl_coupled[3]  # BB
        power[2] = cl_coupled[1]  # EB

        pyfits.writeto(power_file, power, overwrite=True)

        end = time.time()
        print(f"nside={self.nside} for bin pair ({i}, {j}) took {end - start} seconds")
        return

    def run_noise(self, idx):
        print(f"Running pseudo noise power spectrum for nside={self.nside}")
        start = time.time()

        data = self.load_data(idx)
        pixels = hp.ang2pix(self.nside, data["ra"], data["dec"], lonlat=True)

        if self.noise_analytic:
            Nl = self.get_noise_analytic(pixels, data["e1"], data["e2"], data["weight"])
            Nl = Nl * np.ones(self.ell_max)
            noise_file = os.path.join(self.output_folder, f"noise_analytic_{idx}.fits")
            if os.path.exists(noise_file) and not self.overwrite:
                print(f"{noise_file} exists and overwrite is set to False")
            pyfits.writeto(noise_file, Nl, overwrite=True)
        else:
            Nl = self.get_noise_sim(
                pixels, data["e1"], data["e2"], data["e_rms"], data["sigma_e"]
            )
            noise_file = os.path.join(self.output_folder, f"noise_sim_{idx}.fits")
            if os.path.exists(noise_file) and not self.overwrite:
                print(f"{noise_file} exists and overwrite is set to False")
            pyfits.writeto(noise_file, Nl, overwrite=True)

        end = time.time()
        print(
            f"Running pseudo noise power spectrum for nside={self.nside} took {end - start} seconds"
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

    def get_noise_analytic(self, pixels, e1, e2, weight):
        e1_weight = np.bincount(
            pixels, weights=e1**2 * weight**2, minlength=self.npix
        )
        e2_weight = np.bincount(
            pixels, weights=e2**2 * weight**2, minlength=self.npix
        )
        weight_map = np.bincount(pixels, weights=weight, minlength=self.npix)
        e1_weight[weight_map != 0] /= weight_map[weight_map != 0]
        e2_weight[weight_map != 0] /= weight_map[weight_map != 0]
        Nl = 0.5 * (np.mean(e1_weight) + np.mean(e2_weight))
        Nl *= self.Omega_pix
        return Nl

    def get_noise_sim(self, pixels, e1, e2, e_rms, sigma_e, N=100):
        Nl = np.zeros((2, N, self.ell_max))
        for seed in range(N):
            int1, int2, meas1, meas2 = catutil.simulate_shape_noise(
                e1=e1,
                e2=e2,
                e_rms=e_rms,
                sigma_e=sigma_e,
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
            mask, g1_map, g2_map = self.get_map(
                pixels, e1_mock, e2_mock, save_window=False, i=seed
            )
            field = nmt.NmtField(mask, [g1_map, g2_map], spin=2, lite=True)
            cl_coupled = nmt.compute_coupled_cell(field, field)
            Nl[0, seed] = cl_coupled[0]  ## EE
            Nl[1, seed] = cl_coupled[3]  ## BB
        Nl = np.average(Nl, axis=1)
        return Nl


if __name__ == "__main__":
    parser = ArgumentParser(description="simulate blended images")
    parser.add_argument(
        "--nside", type=int, default=128, help="Nside to be used for generating maps"
    )
    parser.add_argument(
        "--power",
        dest="power",
        default=False,
        action="store_true",
        help="whether to save the power spectra (pseudo)",
    )
    parser.add_argument(
        "--noise",
        dest="noise",
        default=False,
        action="store_true",
        help="whether to compute the noise power spectra (pseudo)",
    )
    parser.add_argument(
        "--noise_analytic",
        dest="noise_analytic",
        default=False,
        action="store_true",
        help="whether to compute the noise power spectra (pseudo) analytically",
    )
    parser.add_argument(
        "--save_window",
        dest="save_window",
        default=False,
        action="store_true",
        help="whether to save the window function",
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
    nside = args.nside
    power = args.power
    noise = args.noise
    noise_analytic = args.noise_analytic
    save_window = args.save_window
    auto = args.auto

    idx = np.array([0, 4, 7, 9]) if auto else np.arange(10)

    start = time.time()

    worker = PowerWorker(nside, power, noise, noise_analytic, save_window)
    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    ncores = get_processor_count(pool, args)
    print(f"Running with mpi={args.mpi} with ncores = {ncores}")
    print(
        f"power = {power}, noise = {noise}, noise_analytic = {noise_analytic}, window={save_window}, auto={auto}"
    )

    if power:
        pool.map(worker.run, idx)
    if noise:
        pool.map(worker.run_noise, np.array([0, 1, 2, 3]))

    end = time.time()
    pool.close()

    print(f"Power Spectra took {end - start} seconds")
