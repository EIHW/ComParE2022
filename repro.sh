#!/usr/bin/env -S nix develop path:.envs/default --command bash

#SBATCH --partition=nixos
#SBATCH --time=3-0:0:0
#SBATCH --mem=32G
#SBATCH --get-user-env
#SBATCH --export=ALL
#SBATCH --cpus-per-task=16
#SBATCH -o repro.%A.%a.out
#SBATCH -J stottern

# ----------------------- Fill in information here --------------------------------------------------- #
flags=""
stage=""

# ----------------------- Leave everything below as is ----------------------------------------------- #

# enable crazier wildcards
shopt -s extglob

export PYTHONUNBUFFERED=1

cmd="dvc repro $flags $stage"

echo $cmd
$cmd