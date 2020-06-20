#!/bin/bash
#
#SBATCH --job-name=tennis
#SBATCH --comment="Diese Beschreibung hilft meinen Job zu verstehen"
#SBATCH --begin=20:00
#SBATCH --mem=2048
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<mustafa.yasin@campus.lmu.de>
#SBATCH --chdir=/home/y/yasin/Dokumente/asp/asp
#SBATCH --output=/home/y/yasin/Dokumente/asp_out
#SBATCH --ntasks=1
# ...

#Hier läuft das Hauptprogramm:

echo "Hallo tennis.atp, alles klar!"
pip install -r requirements.txt
python ddpg.py
