#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 01:15:25 2025

@author: dutta26
"""

import torch
from torchvision import transforms
import os
import gan_utils as utils

#Configuration
##############################################################
TEST_DATA_ROOT = "/scratch/bell/dutta26/mvtec/bottle/test/"
GROUND_TRUTH_ROOT = "/scratch/bell/dutta26/mvtec/bottle/ground_truth/"
MODELS_DIR = "gan_models"
OUTPUT_DIR = "gan_analysis_plots" # Directory to save the final plot files

# Model & Search Parameters
IMAGE_SIZE = 256
LATENT_DIM = 100
SEARCH_ITERATIONS = 500 
SEARCH_LR = 0.01
#######################################################################

def main():
    

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #Load Models and Define Transforms ---
    generator, discriminator = utils.load_gan_models(MODELS_DIR, LATENT_DIM, DEVICE)

    eval_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    #TEst two reduction methods to see which one works best
    reductions_to_test = ['mean','sum']

    for reduction in reductions_to_test:
        print("\n" + "="*50)
        print(f"ANALYSIS FOR REDUCTION: '{reduction}'")
        

        title_suffix = f"(Reduction: {reduction})"

        #Calculate scores for the entir dataset
        results_data = utils.calculate_scores_for_dataset(
            generator, TEST_DATA_ROOT, eval_transforms, DEVICE,
            SEARCH_ITERATIONS, SEARCH_LR, reduction=reduction)

        # Prepare data for plotting functions
        y_true = [item['is_defect'] for item in results_data]
        y_scores = [item['anomaly_score'] for item in results_data]

        # Generate and save image-level ROC curve
        roc_path = os.path.join(OUTPUT_DIR, f"image_roc_curve_{reduction}.png")
        utils.plot_roc_curve(y_true, y_scores, roc_path, title_suffix)

        #Generate and save score hitograms
        hist_path = os.path.join(OUTPUT_DIR, f"score_histogram_{reduction}.png")
        utils.plot_score_histograms(results_data, hist_path, title_suffix)

        # Generate and save pixel-level ROC curve
        pixel_roc_path = os.path.join(OUTPUT_DIR, f"pixel_roc_curve_{reduction}.png")
        utils.plot_pixel_roc_curve(
            generator, TEST_DATA_ROOT, GROUND_TRUTH_ROOT, eval_transforms, DEVICE,
            SEARCH_ITERATIONS, SEARCH_LR, reduction, pixel_roc_path, title_suffix
        )


if __name__ == "__main__":
    main()

