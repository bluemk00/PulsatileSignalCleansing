# Pulsatile Signal Cleansing


## Overview

This repository contains the Python code for **Salient Facets in AI for Cleansing Pulsatile Physiological Signals: Knowledge Incorporation, Real-time, Assessments, and Adoption**. Our research focuses on the experimental validation of four aspects for the clinical application of cleansing pulsatile physiological signals.


## Library Dependencies

- Python == 3.8
- numpy == 1.19
- tensorflow == 2.4


## Code Overview and Execution Guide

This repository includes instructions for running two of the metric-based evaluations among the three experimental assessments in our research. You can download the processed data via the link below.

[Download the processed data here](https://www.dropbox.com/scl/fo/lxdqh9p1fqvhip7dcfrbe/AGhSbWSU45kqVVjU0bVWM0o?rlkey=etlsztymn8lyw72nbkt68qgv6&dl=1)



### Model-based Cleansing Performance (MCP)

In the [Experiments/MCP](https://github.com/bluemk00/PulsatileSignalCleansing/tree/main/Experiments/MCP) directory, the cleansing performance on augmented artifacts is evaluated according to different models. Navigate to this folder and run [evaluation.py](https://github.com/bluemk00/PulsatileSignalCleansing/tree/main/Experiments/MCP/evaluation.py) as shown below. You will see that CSV files are generated in the [Experiments/MCP/Results](https://github.com/bluemk00/PulsatileSignalCleansing/tree/main/Experiments/MCP/Results) folder.

```bash
cd Experiments/MCP
python evaluation.py
```


### Model-based Downstream task Performance (MDP)

In the [Experiments/MDP](https://github.com/bluemk00/PulsatileSignalCleansing/tree/main/Experiments/MDP) directory, the cascading effect of cleansing on real artifacts is evaluated by two types of downstream tasks: **IOH Prediction Performance** and **PPG to ABP Transfer Performance**.


#### IOH Prediction Performance

Navigate to the [Experiments/MDP/IOHPrediction](https://github.com/bluemk00/PulsatileSignalCleansing/tree/main/Experiments/MDP/IOHPrediction) directory and run [evaluation_30s60s.py](https://github.com/bluemk00/PulsatileSignalCleansing/tree/main/Experiments/MDP/IOHPrediction/evaluation_30s60s.py) and [evaluation_90s.py](https://github.com/bluemk00/PulsatileSignalCleansing/tree/main/Experiments/MDP/IOHPrediction/evaluation_90s.py) as shown below. You will see that CSV files are generated in the [Experiments/MDP/IOHPrediction/Results](https://github.com/bluemk00/PulsatileSignalCleansing/tree/main/Experiments/MDP/IOHPrediction/Results) folder.

```bash
cd Experiments/MDP/IOHPrediction
python evaluation_30s60s.py
python evaluation_90s.py
```


#### PPG to ABP Transfer Performance

Navigate to the [Experiments/MDP/PPGtoABP](https://github.com/bluemk00/PulsatileSignalCleansing/tree/main/Experiments/MDP/PPGtoABP) directory and run [evaluation.py](https://github.com/bluemk00/PulsatileSignalCleansing/tree/main/Experiments/MDP/PPGtoABP/evaluation.py) as shown below. You will see that CSV file is generated in the [Experiments/MDP/PPGtoABP/Results](https://github.com/bluemk00/PulsatileSignalCleansing/tree/main/Experiments/MDP/PPGtoABP/Results) folder.

```bash
cd Experiments/MDP/PPGtoABP
python evaluation.py
``` 



## Contact Information


- **Email:** [lyjune0070@gmail.com, bluemk00@gmail.com]

- **Institution:** [Department of Cancer AI and Digital Health, Graduate School of Cancer Science & Policy, National Cancer Center, KOREA]



## Acknowledgements

We appreciate the contributions and feedback from all users and collaborators who are helping to improve our research.
