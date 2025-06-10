#!/bin/bash
#SBATCH --job-name=loss_channels
#SBATCH --output=get_loss.out
#SBATCH --error=get_loss.err
#SBATCH --account=pi-liangjiang
#SBATCH --time=35:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=7
#SBATCH --ntasks-per-node=6
#SBATCH --mem-per-cpu=4000
#SBATCH --mail-type=ALL  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mahadevans@rcc.uchicago.edu  # Where to send email

module load python
source activate gkp-env

python get_loss.py