## Multi-domain Transfer Learning for Image Classification

Authors: Ansh Godha, Divya Talesra

## Description

This repository is home to a series of experiments run by us to try out strategies for enhancing learning across multiple domains. It is known that transfer learning across extreme domain transfers often yields poor results - for instance, pre-training a network on a series of images of animals would likely yield poor results when trying to classify satellite images. We will test combining results from separately trained models, so this gives us slightly more variation in our datasets. This is similar to what datasets like ImageNet provide, except that we will pre-train domain-specific models separately instead of pre-training on a dataset with mixed classes. Our experiment will determine if our method will lead to improvements.

## Setting up

Once you clone the repository, run `./setup.sh` (tested on MacOS). This is a script to download the datasets and store them in the `datasets/` directory, which is gitignored because of its size. This script also runs another utility script in `utils` to set up the annotations csvs for PyTorch.
