#!/bin/bash
#

# Name your file
#SBATCH --job-name=tennis

# Leave a comment about your job 
#SBATCH --comment="DDPG Model"

# Time when you job needs to start 
#SBATCH --begin=14:00
#SBATCH --mem=2048
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL

# Leave an email to receive a feedback once your job is completed
#SBATCH --mail-user=<mustafa.yasin@campus.lmu.de>

# The dir where your project is located
#SBATCH --chdir=/home/y/yasin/Dokumente/asp/asp

# The dir where the output should be located
#SBATCH --output=/home/y/yasin/Dokumente/asp_out
#SBATCH --ntasks=1
# ...

# Here is the section where the main programm runs:

echo "Hallo tennis.atp, alles klar!"
pip install -r requirements.txt
python ddpg.py
