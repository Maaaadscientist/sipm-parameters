import numpy as np
import pandas as pd
import math
import os
import time
import scipy.stats as stats
import matplotlib.pyplot as plt
import ROOT
import argparse  # Import argparse module

PHOTONS_PER_TEST = 9846  # Number of photons simulated in each test

single_pe_resolution = 0.15
# Example parameters
#dcr = 50.
#PTE = 0.836  # Probability of photon hitting SiPM active area 815074 / 975251
PTE = 0.8  # Probability of photon hitting SiPM active area 1601665 / 2000000
transmittance = 0.9
PTE_ir = 0.75
corr_factor = 2.05 * PTE_ir # arxiv 2312.12901


keys = [round(2.0 + 0.1 * i, 1) for i in range(51)]

#optical_crosstalk_param = 0.2  # Parameter for generalized Poisson distribution
pde_dict = {"max": 0.44, "typical":0.47}
pct_dict = {"max": 0.15, "typical":0.12}
dcr_dict = {"max": 41.7, "typical":13.9}
pap_dict = {"max": 0.08, "typical":0.04}
gain_dict = {"max": 1e6, "typical":4e6}


def parse_arguments():
    parser = argparse.ArgumentParser(description='Photon Simulation Parameters')
    parser.add_argument('--N', type=int, default=20, help='Number of tests')
    parser.add_argument('--energy', type=float, default=1.0, help='deposit energy (MeV)')
    parser.add_argument('--ov', type=float, default=3.0, help='Overvoltage, for HPK test: -1: max, -2: typical')
    parser.add_argument('--level', type=int, required=True, help='Random level: 1: whole detector, 2: by tile, 3: by channel, 4: by channel and different ov ')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, default='all_hist.root', help='Output ROOT file path')
    parser.add_argument('--seed', type=int,default=123456, help='Random seed for simulation', required=False)
    args = parser.parse_args()
    return args.N, args.energy, args.ov, args.level, args.input, args.output, args.seed

# Parse command line arguments
N, energy, ov, level, input_file, output_file, seed = parse_arguments()
# Initialize random seed
if seed is not None:
    np.random.seed(seed)

f1 = ROOT.TFile(input_file)

def clear_screen():
    # For Windows
    if os.name == 'nt':
        _ = os.system('cls')
    # For macOS and Linux (here, os.name is 'posix')
    else:
        _ = os.system('clear')

def get_hist(f1, column_name):
    hist = f1.Get(column_name)
    return hist

def average_value_from_root(f1, column_name, selection_criteria=None):
    # Check if selection criteria are provided
    hist = f1.Get(column_name)
    
    return hist.GetMean()

# Function to get a specific value from a row based on tsn, ch, and column name
def get_value_by_tsn_ch(df, reference_dict, tsn, ch, column_name):
    row_index = reference_dict.get((tsn, ch))
    if row_index is not None and column_name in df.columns:
        return df.at[row_index, column_name]
    else:
        return "Value not found"

def generalized_poisson_pmf(k, mu, lambda_):
    exp_term = np.exp(-(mu + k * lambda_))
    main_term = mu * ((mu + k * lambda_) ** (k - 1))
    factorial_term = math.factorial(k)
    return (main_term * exp_term) / factorial_term

def generate_random_generalized_poisson(mu, lambda_, max_k=35):
    # Generate PMF values
    pmf_values = [generalized_poisson_pmf(k, mu, lambda_) for k in range(max_k)]

    # Normalize the PMF
    total = sum(pmf_values)
    normalized_pmf = [value / total for value in pmf_values]

    # Random sampling based on the PMF
    return np.random.choice(range(max_k), p=normalized_pmf)

def borel_pmf(k, lambda_):
    return (lambda_ * k)**(k - 1) * np.exp(-k * lambda_) / math.factorial(k)


def generate_random_borel(lambda_, max_k=20):
    # Generate PMF values
    pmf_values = [borel_pmf(k+1, lambda_) for k in range(max_k)]

    # Normalize the PMF
    total = sum(pmf_values)
    normalized_pmf = [value / total for value in pmf_values]

    # Random sampling based on the PMF
    return np.random.choice(range(max_k), p=normalized_pmf)

def simulate_dcr(ov_str):
    pde_name = "pde_" + ov_str
    pct_name = "lambda_" + ov_str
    dcr_name = "dcr_" + ov_str
    gain_name = "gain_abs_" + ov_str
    ap_name = "ap_mean_" + ov_str

    dcr = average_value_from_root(f1, dcr_name)
    # 360ns, 4024 tiles, with each tile having 32 6mm*12mm chips
    init_pe = dcr * 360 * 1e-9 * (4024 * 32 * 6 * 12) 
    pe = np.random.poisson(init_pe)
    lambda_hist = get_hist(f1, pct_name)
    pe_ct = pe
    for _ in range(int(pe)):
        lambda_ = lambda_hist.GetRandom()* corr_factor
        pe_ct += generate_random_borel(lambda_)

    pe_ct_air = pe
    for _ in range(int(pe)):
        lambda_air = lambda_hist.GetRandom()
        pe_ct_air += generate_random_borel(lambda_air)
    # Simulate afterpulsing
    
    ap_hist = get_hist(f1, ap_name)
    pe_ap = pe_ct
    for _ in range(int(pe_ct)):
        pap = ap_hist.GetRandom()
        pe_ap += np.random.normal(loc=pap, scale=pap)

    # Simulate gain
    gain_hist = get_hist(f1, gain_name)
    gain = 0

    mean_gain = gain_hist.GetMean()
    for _ in range(int(pe_ap)):
        gain += gain_hist.GetRandom() / mean_gain

    smear_gain = 0
    for _ in range(int(pe_ap)):
        smear_gain += np.random.normal(1, single_pe_resolution)

    corr_ct = pe_ct * (1-lambda_hist.GetMean() * corr_factor)
    corr_air_ct = pe_ct_air * (1-lambda_hist.GetMean())
    corr_ap = (pe_ap - pe_ct * ap_hist.GetMean()) * (1-lambda_hist.GetMean()* corr_factor )
    corr_q  = (gain - pe_ct * ap_hist.GetMean()) * (1-lambda_hist.GetMean()* corr_factor)
    corr_qres  = (smear_gain - pe_ct * ap_hist.GetMean()) * (1-lambda_hist.GetMean()* corr_factor)
    return init_pe, pe, corr_ct, corr_ap, corr_q , corr_qres, corr_air_ct
        
def main():
    init_photons = int(PHOTONS_PER_TEST * energy)
    Nbins = init_photons*2
    # Data collection arrays
    h_init = ROOT.TH1F("hist_init", "hist_init", Nbins,0,Nbins)
    h_LS = ROOT.TH1F("hist_LS", "hist_LS", Nbins,0,Nbins)
    h_pte = ROOT.TH1F("hist_PTE", "hist_PTE", Nbins,0,Nbins)
    h_pde = ROOT.TH1F("hist_PDE", "hist_PDE", Nbins,0,Nbins)
    h_ct = ROOT.TH1F("hist_ct", "hist_ct", Nbins,0,Nbins)
    h_ct_air = ROOT.TH1F("hist_ct_air", "hist_ct_air", Nbins,0,Nbins)
    h_ap = ROOT.TH1F("hist_ap", "hist_ap", Nbins,0,Nbins)
    h_dcr = ROOT.TH1F("hist_dcr", "hist_dcr", Nbins,0,Nbins)
    h_charge = ROOT.TH1F("hist_charge", "hist_charge", Nbins,0,Nbins)
    h_qres = ROOT.TH1F("hist_qres", "hist_qres", Nbins,0,Nbins)
    hist_dcr_init = ROOT.TH1F("dcr_init", "dcr_init", 2000,0,2000)
    hist_dcr_poisson = ROOT.TH1F("dcr_poisson", "dcr_poisson", 2000,0,2000)
    hist_dcr_ct = ROOT.TH1F("dcr_ct", "dcr_ct", 2000,0,2000)
    hist_dcr_ap = ROOT.TH1F("dcr_ap", "dcr_ap", 2000,0,2000)
    start_time = time.time()
    for i in range(N):
        
        # Refresh output every X events (e.g., every 10 events)
        if (i + 1) % 100 == 0 or i == N - 1:
            elapsed_time = time.time() - start_time
            percent_complete = ((i + 1) / N) * 100
            avg_time_per_event = elapsed_time / (i + 1)
            estimated_total_time = avg_time_per_event * N
            estimated_remaining_time = estimated_total_time - elapsed_time

            clear_screen()
            print(f"Event {i + 1}/{N} complete")
            print(f"Completion: {percent_complete:.2f}%")
            print(f"Elapsed Time: {elapsed_time:.2f} seconds")
            print(f"Estimated Total Time: {estimated_total_time:.2f} seconds")
            print(f"Estimated Time Remaining: {estimated_remaining_time:.2f} seconds")
            time.sleep(0.01)  # Small delay to ensure the screen refresh is noticeable

        LS_photons = np.random.poisson(init_photons)
        dcr_results = simulate_dcr(str(round(ov, 1)).replace(".", "_")) 
        dcr_init = dcr_results[0]
        dcr_poisson_pe = dcr_results[1]
        dcr_pe_ct  =dcr_results[2] 
        dcr_pe_ctap= dcr_results[3]
        dcr_charge=dcr_results[4]
        dcr_qres=dcr_results[5]
        dcr_ct_air=dcr_results[6]
        hist_dcr_init.Fill(dcr_init)
        hist_dcr_poisson.Fill(dcr_poisson_pe)
        hist_dcr_ct.Fill(dcr_pe_ct)
        hist_dcr_ap.Fill(dcr_pe_ctap)
        hit_photons = np.random.binomial(LS_photons, PTE) 
        PDE = average_value_from_root(f1, f"pde_{ov:.1f}".replace(".","_")) / transmittance # PTE removed
        detected_photons = np.random.binomial(hit_photons, PDE)
        #initial_pe = detected_photons + dcr_poisson_pe - dcr_init
        initial_pe = detected_photons + dcr_qres - dcr_init
        ct_pe = detected_photons
        lambda_hist = get_hist(f1, f"lambda_{ov:.1f}".replace(".","_"))
        for i in range(int(detected_photons)):
            lambda_ = lambda_hist.GetRandom() * corr_factor # corrected for external CT
            ct_pe += generate_random_borel(lambda_)
        #corr_ct = ct_pe * (1 - lambda_hist.GetMean() * corr_factor) + dcr_pe_ct - dcr_init
        corr_ct = ct_pe * (1 - lambda_hist.GetMean() * corr_factor) + dcr_qres - dcr_init

        ct_air_pe = detected_photons
        for i in range(int(detected_photons)):
            lambda_air = lambda_hist.GetRandom() # corrected for external CT
            ct_air_pe += generate_random_borel(lambda_air)
        #corr_air_ct = ct_air_pe * (1 - lambda_hist.GetMean()) + dcr_ct_air - dcr_init
        corr_air_ct = ct_air_pe * (1 - lambda_hist.GetMean()) + dcr_qres - dcr_init

        ap_pe = ct_pe
        ap_hist = get_hist(f1, f"ap_mean_{ov:.1f}".replace(".","_"))
        for i in range(int(ct_pe)):
            pap = ap_hist.GetRandom()
            ap_pe += np.random.normal(loc=pap, scale=pap)
        #corr_ap = (ap_pe - ct_pe * ap_hist.GetMean()) * (1 - lambda_hist.GetMean() * corr_factor) + dcr_pe_ctap - dcr_init
        corr_ap = (ap_pe - ct_pe * ap_hist.GetMean()) * (1 - lambda_hist.GetMean() * corr_factor) + dcr_qres - dcr_init

        gain = 0
        gain_hist = get_hist(f1, f"gain_abs_{ov:.1f}".replace(".","_"))
        mean_gain = gain_hist.GetMean()
        for _ in range(int(ap_pe)):
            gain += gain_hist.GetRandom() / mean_gain
        #corr_q = (gain - ct_pe * ap_hist.GetMean()) * (1 - lambda_hist.GetMean() *  corr_factor) + dcr_charge - dcr_init
        corr_q = (gain - ct_pe * ap_hist.GetMean()) * (1 - lambda_hist.GetMean() *  corr_factor) + dcr_qres - dcr_init
            
        
        smear_gain = 0
        for _ in range(int(ap_pe)):
            smear_gain += np.random.normal(1, single_pe_resolution)
        
        corr_qres  = (smear_gain - ct_pe * ap_hist.GetMean()) * (1-lambda_hist.GetMean() * corr_factor) + dcr_qres - dcr_init
            
        h_init.Fill(PHOTONS_PER_TEST * energy)
        h_LS.Fill(LS_photons)
        h_pte.Fill(hit_photons / PTE)
        h_pde.Fill(detected_photons / (PTE * PDE))
        h_dcr.Fill(initial_pe/ (PTE * PDE) )
        h_ct.Fill(corr_ct/ (PTE * PDE))
        h_ct_air.Fill(corr_air_ct/ (PTE * PDE))
        h_ap.Fill(corr_ap/ (PTE * PDE))
        h_charge.Fill(corr_q/ (PTE * PDE))
        h_qres.Fill(corr_qres/ (PTE * PDE))
    
    # At the end, where the ROOT file is saved
    f_output = ROOT.TFile(output_file, "recreate")
    h_init.Write()
    h_LS.Write()
    h_pte.Write()
    h_pde.Write()
    h_dcr.Write()
    h_ct.Write()
    h_ct_air.Write()
    h_ap.Write()
    h_charge.Write()
    h_qres.Write()
    hist_dcr_init.Write()
    hist_dcr_poisson.Write()
    hist_dcr_ct.Write()
    hist_dcr_ap.Write()
    f_output.Close()

    f1.Close()
if __name__ == "__main__":
    main()
