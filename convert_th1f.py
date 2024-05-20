import ROOT
import yaml
import numpy as np

# Load config file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Function to get binning info for a given parameter
def get_binning_info(param_name):
    if param_name in config:
        return config[param_name]
    else:
        # Default binning method if not specified in config
        return {"lower_bound": None, "upper_bound": None, "bin_width": None}

# Function to create histogram for a given branch
def create_histogram(tree, branch_name_root, ov):
    branch_name = f"{branch_name_root}_{ov.replace('.', '_')}"
    
    # Get binning info for this parameter
    binning_info = get_binning_info(branch_name_root)
    
    # Extract the data from the branch
    values = []
    for entry in tree:
        values.append(getattr(entry, branch_name))
    
    # Create histogram
    if binning_info["lower_bound"] is not None and binning_info["upper_bound"] is not None and binning_info["bin_width"] is not None:
        # Fixed binning
        num_bins = int((binning_info["upper_bound"] - binning_info["lower_bound"]) / binning_info["bin_width"])
        hist = ROOT.TH1F(branch_name, f"Histogram of {branch_name}", num_bins, binning_info["lower_bound"], binning_info["upper_bound"])
    else:
        # Automatic binning
        hist = ROOT.TH1F(branch_name, f"Histogram of {branch_name}", 100, min(values), max(values))
    
    # Fill histogram
    for value in values:
        hist.Fill(value)
    
    return hist

# Open the ROOT file
input_file = ROOT.TFile("sipm-params.root")
output_file = ROOT.TFile("histograms.root", "RECREATE")

# Get the TTree
tree = input_file.Get("tree")

ov_list = [str(round(ov,1)) for ov in np.arange(2.0,7.1,0.1)]
# Iterate over branches
for branch_name_root in config.keys():
    for ov in ov_list:
        hist = create_histogram(tree, branch_name_root, ov)
        hist.Write()

# Close files
input_file.Close()
output_file.Close()

