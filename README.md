# Autonome Systeme Praktikum
[Python](https://www.python.org/downloads/) 3.6.10 is required to run this project

#Suggestion: using conda env/venv
Install Anacnoda on [Linux](https://docs.anaconda.com/anaconda/install/linux/)\
Install Anaconda on [MacOS](https://docs.anaconda.com/anaconda/install/mac-os/)

Create a virtual Environmet via Anaconda\
`conda create -n yourenvname python=3.6.10 anaconda`

Activate your Environment\
`source activate yourenvname`

Install the all needed dependencies\
`pip install -r requirements.txt`

Start with the traning\
`sbatch train.sh`

### Notice
the version of mlagents is 0.4.0. Source code: (https://github.com/Unity-Technologies/ml-agents/releases/tag/0.4.0)

## Environment:
![Tennis](img/tennis.png)
* Set-up: Two-player game where agents control rackets to bounce ball over a net. 
* Goal: The agents must bounce ball between one another while not dropping or sending ball out of bounds.
* Agents: The environment contains two agent linked to a single brain named TennisBrain. 
* Agent Reward Function (independent): 
    * +0.1 To agent when hitting ball over net.
    * -0.1 To agent who let ball hit their ground, or hit ball out of bounds.
* Brains: One brain with the following observation/action space.
    * Vector Observation space: (Continuous) 8 variables corresponding to position and velocity of ball and racket.
    * Vector Action space: (Continuous) Size of 2, corresponding to movement toward net or away from net, and jumping.
* Reset Parameters: One, corresponding to size of ball.
* Benchmark Mean Reward: 2.5 (means the ball should be hitted about 50 times in a game.)

## Goal
1. **Mean Reward**: should be better than 2.5
2. **Hyperparmeter Tuning** and possibly modify NN structure
3. **Evaluation**: vs. default PPO
4. **Result**: should be save as *figure*
 
## Groupmembers
[Mustafa Yasin](https://github.com/MustafaYasin)\
[Xingjian Chen](https://github.com/marcchan)\
[Yang Mao](https://github.com/leo-mao)\
[Steffen Brandenburg](https://github.com/SteffenBr)
