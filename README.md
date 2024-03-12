# Job Shop Scheduling Problem (JSSP)

## Main
- learning target generation
  - opt/opt_policy_gen.py
  
- imitation learning
  - agent/agent.BC.py
    
- performance evaluation
  - agent/evaluation.py
  - agent/evaluation_dyn.py
    

## Method
- Entire process for determining an assignment
<img src = ./images/entire.png width=100%>

- Graph attention network(GAT) structure to compute the selection probability of each node 
<img src = ./images/GAT.png width=80%>

- Sequential decision process with multiple transitions 
<img src = ./images/transition.png width=100%>


## Packages
- CUDA 11.6
  - conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
    
- torch geometric 2.1.0
  - conda install pyg -c pyg
  
- ortools 9.6.2534
  - pip install ortools
  
- plotly 5.9.0
  - conda install plotly
    
- matplotlib 3.7.1
  - conda install matplotlib
  
- pandas 1.5.3
  - conda install pandas
    
- pickle 1.0.2
  - pip install pickle-mixin
    
- networkx 2.8.4
  - conda install networkx
