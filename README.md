# Autonome Systeme Praktikum
[Python](https://www.python.org/downloads/) >= 3.6.1 is required to run this project

Login into your cip virtual machine\
`ssh username@remote.ifi.lmu.de`

**virtual env by conda**  
Install Anaconda on [Linux](https://docs.anaconda.com/anaconda/install/linux/)
```
curl -L https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh > install.sh
chmod +x install.sh
./install.sh`
```
Create a virtual environment and install dependency via Anaconda \
`conda create --name asp --file requirements.txt`

Change the path in both `train.sh` and `tennis.py` then start with the traning\
`sbatch train.sh`

Checkout train progress by `squeue -u your_username`, checkout output by `cat run.log`

or you can use **virtualenv**, but you have to install the env manually and change the `conda activate` into `source activate`
in `train.sh`

### Groupmembers
[Mustafa Yasin](https://github.com/MustafaYasin)  
[Xingjian Chen](https://github.com/marcchan)  
[Yang Mao](https://github.com/leo-mao)  
[Steffen Brandenburg](https://github.com/SteffenBr)
