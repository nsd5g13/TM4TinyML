# TM4TinyML: Tsetlin Machine and Compressed Tsetlin Machine for TinyML

<!-- ABOUT THE PROJECT -->
## About The Project

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

Install pyTsetlinMachine (for Tsetlin Machine training, original source code available at [https://github.com/cair/pyTsetlinMachine](https://github.com/cair/pyTsetlinMachine)):
   ```sh
   cd pyTsetlinMachine
   python3 setup.py install
   ```

Install Larq (an open-source Python library for Binarized Neural Network, available at [https://docs.larq.dev/larq/](https://docs.larq.dev/larq/))
- Used versions: python==3.7.13, tensorflow=2.0.0

Build the firmware for a given board to support MicroPython, related documentation available at [https://github.com/micropython/micropython/tree/master](https://github.com/micropython/micropython/tree/master)
- Used micron-controller board: STM32F746G-DISCO

## Usage

### Booleanization

A Tsetlin machine (TM) requires all raw features of a dataset represented in the form of Bool. We provide source code to booleanize multiple TinyML open source datasets:
- [EMG](https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures)
- [Gas sensor array drift](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset)
- [Gesture phase segmentation](https://archive.ics.uci.edu/ml/datasets/gesture+phase+segmentation)
- [Human activity](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- [Mammographic mass](http://archive.ics.uci.edu/ml/datasets/mammographic+mass)
- [Sensorless drive diagnosis](https://archive.ics.uci.edu/ml/datasets/dataset+for+sensorless+drive+diagnosis) 
- [Sport activity](https://archive.ics.uci.edu/ml/datasets/Daily+and+Sports+Activities)
- [Statlog (vehicle silhouette)](https://archive.ics.uci.edu/dataset/149/statlog+vehicle+silhouettes)

Before Booleanizing, download all raw datasets and put the dataset directory at "/TinyML_raw_dataset/".
   ```sh
   cd DIET/Booleanization
   python3 booleanization.py [dataset_name]
   ```
Options for [dataset_name] are emg, gas, gesture, har, mammography, sensorless, sports, statlog and all, where "all" suggests producing Booleanized datasets for all above.

### Vanilla TM Training

Source code to train a vanilla TM-based model is located in DIET/VanillaTM_training.

   ```sh
   usage: StandardTraining.py clauses T s epochs budget dataset_name

 positional arguments:
     clauses         Provide the number of clauses per class
     T               Provide the value of "Threshold"
     s               Provide the value of "Strength" for literal include
     epochs          Proivde the number of training epochs
     budget          Provide the constrain for the maximal number of literals included in each clause
     dataset_name    Provide the name of the dataset. Options include emg, gas, gesture, har, mammography, sensorless, sports and statlog
   ```

Example:
   ```sh
   cd DIET/VanillaTM_training
   python3 StandardTraining.py 300 14 7.5 200 320 emg
   ```

### Iterative Training-Excluding for Compressed TM

Source code to use DIET to train a compressed TM is located in DIET/DIET_training.

   ```sh
   usage: diet.py clauses T s epochs epochs_every_exclude budget dataset_name

 positional arguments:
     clauses               Provide the number of clauses per class
     T                     Provide the value of "Threshold"
     s                     Provide the value of "Strength" for literal include
     epochs                Proivde the total number of training epochs
     epochs_every_exclude  Specify the number of epochs after each excluding process  
     budget                Provide the constrain for the maximal number of literals included in each clause (we do not constrain the model in the experiment, so the budget is set according the number of literals for each dataset)
     dataset_name          Provide the name of the dataset. Options include emg, gas, gesture, har, mammography, sensorless, sports and statlog
   ```

Example:
   ```sh
   cd DIET/DIET_training
   python3 diet.py 300 14 7.5 200 5 320 emg
   ```

Suggested hyperparameters:
| Dataset | Clauses | T | s | epochs | epochs_every_exclude |
| ------- | ------- | - | - | ------ | -------------------- |
| emg	  | 300	    | 14|7.5| 200    | 3		    |
| gas	  | 300	    | 12|10 | 200    | 5		    |
| gesture | 500     | 25|9  | 250    | 25                   |
| har     | 200     | 14|6  | 250    | 1                    |
| mammograpy| 50    | 7 |3  | 100    | 1                    |
| sensorless| 300   | 15| 10| 100    | 4		    |
| sports  | 150     | 12| 4 | 50     | 1                    |
| statlog | 300     | 16| 3 | 100    | 1                    |

### MicroPython Implementation for On-board Inference

Source code to export TM models for MicroPython-based on-board inference is located at TMtoMicropython/RedressGen.

Example:
   ```sh
   cd TMtoMicropython
   python3 RedressGen.py vanillaORdiet
   ```
Relace "vanillaORdiet" by "vanilla" or "diet" to generate TM models for vanilla or compressed TM, respectively.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
