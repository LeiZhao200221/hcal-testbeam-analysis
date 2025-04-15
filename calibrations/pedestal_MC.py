#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy import stats
import os
import argparse

# \033[92m
# Code Idea:
# - Read CSV file with two ADC endpoints (adc_sum_end0, adc_sum_end1).
# - Process each endpoint by labeling (end 0 or 1) and renaming the ADC column.
# - Combine into a long-format DataFrame.
# - Group by 'layer', 'strip', and 'end'; for each group:
#     * Generate histogram, find the mode bin,
#     * Use the bin center ±200 as the fitting range,
#     * Fit a Gaussian to extract the pedestal and standard deviation.
# - Save the result DataFrame as a CSV file without the row indices.


def process_csv(input_file: str, output_folder: str):
    output_file = os.path.join(output_folder, "MC_pedestal.csv")
    df = pd.read_csv(input_file)

    # Process endpoint 0
    df_long0 = df[['layer', 'strip', 'adc_sum_end0']].copy()
    df_long0['end'] = 0
    df_long0 = df_long0.rename(columns={'adc_sum_end0': 'adc_sum'})
    
    # Process endpoint 1
    df_long1 = df[['layer', 'strip', 'adc_sum_end1']].copy()
    df_long1['end'] = 1
    df_long1 = df_long1.rename(columns={'adc_sum_end1': 'adc_sum'})
    
    # Combine data
    df_long = pd.concat([df_long0, df_long1], ignore_index=True)
    
    results = []
    bin_width = 20
    fit_range_half_width = 200
    
    grouped = df_long.groupby(['layer', 'strip', 'end'])
    for (layer, strip, end), group in grouped:
        adc_values = group['adc_sum'].dropna().values
        if len(adc_values) == 0:
            continue
        
        min_val = adc_values.min()
        max_val = adc_values.max()
        bins = np.arange(min_val, max_val + bin_width, bin_width)
        if len(bins) < 2:
            continue
        
        counts, bin_edges = np.histogram(adc_values, bins=bins)
        mode_index = np.argmax(counts)
        bin_center = (bin_edges[mode_index] + bin_edges[mode_index+1]) / 2.0
        
        low_limit = bin_center - fit_range_half_width
        high_limit = bin_center + fit_range_half_width
        adc_fit = adc_values[(adc_values >= low_limit) & (adc_values <= high_limit)]
        if len(adc_fit) == 0:
            continue
        
        pedestal, std_dev = stats.norm.fit(adc_fit)
        
        results.append({
            'layer': layer,
            'strip': strip,
            'end': end,
            'pedestal': pedestal,
            'std_dev': std_dev
        })
    
    df_result = pd.DataFrame(results)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    df_result.to_csv(output_file, index=False)  # Save the result DataFrame as a CSV file without the row indices
    print("Pedestal results saved to:", output_file)

def main():
    parser = argparse.ArgumentParser(
        description="Process a CSV file to extract pedestal info and perform Gaussian fitting."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Full path to the input CSV file with columns 'layer', 'strip', 'adc_sum_end0', 'adc_sum_end1'.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to the output folder for saving the result CSV file.")
    
    args = parser.parse_args()
    process_csv(args.input, args.output)

if __name__ == "__main__":
    main()
