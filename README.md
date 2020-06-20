# Autonome Systeme Praktikum
[Python](https://www.python.org/downloads/) 3.6.0 is required to run this project

Login into your cip virtual machine\
`ssh username@remote.ifi.lmu.de`

Install Anacoda [Linux](bash ~/Downloads/Anaconda3-2020.02-Linux-x86_64.sh) [MacOS](https://docs.anaconda.com/anaconda/install/mac-os/)\

Create a virtual Environmet via Anaconda\
`conda create -n yourenvname python=3.6 anaconda`

Activate your Environment\
`source activate yourenvname`

Install the all needed dependencies\
`pip install -r requirements.txt`

Configur the slurm script (asp.sh) with your credentials and run it.\
`sbatch -p All asp.sh`

Start with the traning\
`python train.py`


### Groupmemebers
[Mustafa Yasin](https://github.com/MustafaYasin)\
[Xingjian Chen](https://github.com/marcchan)\
[Yang Mao](https://github.com/leo-mao)\
[Steffen Brandenburg](https://github.com/SteffenBr)
