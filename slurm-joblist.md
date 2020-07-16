
#slurm train history

jobid | nn config|                                           target goal | finished epoch | folder | best result
---| ---| ---| --- | --- | ---
242776 | nn with bn dp 20k episodes, default hyperparameter | 0.7 | 612 episodes  |  |
242815 | nn with bn dp 20k episodes, default hyperparameter | 0.9 | 853 episodes  |  |
242889 | nn with bn dp 20k episodes, default hyperparameter | 0.95| 1261 episodes | /0708_16:35  | 
242911 | nn with bn dp 20k episodes, default hyperparameter | 2.5 | 2k, not enough|/0708_17:11   | 
242912 | nn with bn dp 20k episodes, default hyperparameter | 3   | 2k,not enough | /0708_17:24  |


---0709--- 

jobid | nn config|                                           target goal | finished epoch | folder | best result        
---| ---| ---| --- | --- | ---                                                                                          
243684 | nn with bn dp 20k episodes, default hypoparameter | 3 | ? | /0709_04:26|
243805 | default nn, default hypoparameter | 3 | ? | /0709_11:15 |


---0711---


jobid | nn config|                                           target goal | finished epoch | folder | best result        
---| ---| ---| --- | --- | ---                                                                                          
244927 | default nn, default hypoparameter | 3 | ? | /0710_20:17 | 2150:0.91 |
244928 | changed actor, default critic, default hyperparameter | 3 | ? | /0710_20:22  | --2695:0.94 - (2700)


---0715---


jobid | nn config|                                           target goal | finished epoch | folder | best result | based on        
---| ---| ---| --- | --- | ---                                    
248201 | changed actor, default critic, default hyperparameter, noise * 5% | 3 |   | /0714_18:39 | 1250--0.95, 1280--1.03 | 0712_04:54/a-c_2700.pth
248367| changed actor, default critic, ... noise * 0.1% | 3 | / 