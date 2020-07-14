# Autonome Systeme Praktikum
[Python](https://www.python.org/downloads/) 3.6.10 is required to run this project\
[mlagents](https://github.com/Unity-Technologies/ml-agents/releases/tag/0.4.0) 0.4.0 is required to run this project
## Suggestion: using conda env/venv
Install Anacoda on [Linux] (https://docs.anaconda.com/anaconda/install/linux/)\
Install Anaconda on [MacOS] (https://docs.anaconda.com/anaconda/install/mac-os/)

Create a virtual Environmet via Anaconda\
`conda create -n yourenvname python=3.6.10 anaconda`

Activate your Environment\
`source activate yourenvname`

Install the all needed dependencies\
`pip install -r requirements.txt`

Start with the traning\
`sbatch train.sh`


# Project Introduction

![Tennis](img/tennis-demo.gif)

* **Set-up**: Two-player game where agents control rackets to bounce ball over a net. 
* **Goal**: The agents must bounce ball between one another while not dropping or sending ball out of bounds.
* **Agents**: The environment contains two agent linked to a single brain named TennisBrain. 
* **Agent Reward Function (independent)**: 
    * +0.1 To agent when hitting ball over net.
    * -0.1 To agent who let ball hit their ground, or hit ball out of bounds.
* **Brains**: One brain with the following observation/action space.
    * Vector Observation space: (Continuous) 8 variables corresponding to position and velocity of ball and racket.  
        * `shape`: `(2, 24)` means the env has 2 agents, Each observes a state with length 24. 
        * `obs[0]`: the observation of the left side player 
        * `obs[1]`: the observation of the right side player.
        * take `obs[0]` for example, because the `obs[1]` :
            * `obs[0][16]`:  Relative position of left racket and field in X axis
            * `obs[0][17]`:  Relative position of left racket and field in Y axis
            * `obs[0][18]`:  Velocity of the left racket in X axis
            * `obs[0][19]`:  Velocity of the left racket in Y axis
            * `obs[0][20]`:  Relative position of ball and field in X axis
            * `obs[0][21]`:  Relative position of ball and field in Y axis
            * `obs[0][22]`:  Velocity of the ball in X axis
            * `obs[0][23]`:  Velocity of the ball in Y axis
    
        * `obs[0][8]` - `obs[0][15]`: the observation of the previous step for left agent.
        * `obs[0][0]` - `obs[0][7]`: the observation of the 1 step before the previous step for left agent.

        * source code: `ml-agents-0.4.0/unity-environment/Assets/ML-Agents/Examples/Tennis/Scripts/TennisAgent.cs`: 
        
            ```java
                public override void CollectObservations(){
                    AddVectorObs(invertMult * (transform.position.x - myArea.transform.position.x));
                    AddVectorObs(transform.position.y - myArea.transform.position.y);
                    AddVectorObs(invertMult * agentRb.velocity.x);
                    AddVectorObs(agentRb.velocity.y);
                    AddVectorObs(invertMult * (ball.transform.position.x - myArea.transform.position.x));
                    AddVectorObs(ball.transform.position.y - myArea.transform.position.y);
                    AddVectorObs(invertMult * ballRb.velocity.x);
                    AddVectorObs(ballRb.velocity.y);
                 }
            ```
                      
    * Vector Action space: (Continuous) Size of 2, corresponding to movement toward net or away from net, and jumping.
         * `shape`: `(2, 2)`  the env has 2 agents, 
         * `vectorAction[0]`: action for left side player
         * `vectorAction[1]`: action for right side player  
         * take the `vectorAction[0]` for example:
             * `vectorAction[0][0]`: move value in x axis for left side player, value should be between -1 and 1
             * `vectorAction[0][1]`: move value in Y axis for left side player, value should be between -1 and 1
 
 
     
## Training
### Algorithm: *DDPG*
TODO


### Description:
* Max Reward:the max reward in a episode, for example: `max Reward = 2.5` that means the reward of two players in a game = [2.5, 2.4] or [2.4, 2.5]
* Score: a list contain the max Reward in each game, could be saved in "pth" File, and load to see the change in plot.
* Total Average Score: calculated from the average scores of last 100 episodes, we take it as an indicator for the training

## Goal
1. **Default Train**: Train with DDPG 
2. **Hyperparmeter Tuning** and possibly modify NN structure
3. **Evaluation**: changed Version vs. default DDPG Version
4. **Result**: should be save as *figure*    
    

### Groupmembers
[Mustafa Yasin](https://github.com/MustafaYasin)\
[Xingjian Chen](https://github.com/marcchan)\
[Yang Mao](https://github.com/leo-mao)\
[Steffen Brandenburg] (https://github.com/SteffenBr)
