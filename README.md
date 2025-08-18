# Deep Learning-based Time Series Forecasting for Industrial Discrete Process Data

This repository contains the dataset and source code accompanying the publication available at  https://ieeexplore.ieee.org/document/11087869 .

## Dataset

The dataset is available in the subfolder `dataset/`.
It is a CSV-file, organized as follows:

| **Col.** | **Name**    | **Description**                      | **Type**  | **Symbol**        | **Unit** |
|----------|-------------|--------------------------------------|-----------|-------------------|----------|
| 0        | Time        | Elapsed time since measurement start | --        | $t$               | s        |
| 1        | Voltage0    | DC-motor 0 voltage                   | output    | $U_0$             | V        |
| 2        | Voltage1    | DC-motor 1 voltage                   | output    | $U_1$             | V        |
| 3        | Yaw         | Actual yaw angle                     | measured  | $\varPsi$         | rad      |
| 4        | Pitch       | Actual pitch angle                   | measured  | $\varTheta$       | rad      |
| 5        | TargetYaw   | Target yaw angle                     | input     | $\varPsi_T$       | rad      |
| 6        | TargetPitch | Target pitch angle                   | input     | $\varTheta_T$     | rad      |
| 7        | YawDot      | Yaw, angular velocity                | estimated | $\dot{\varPsi}$   | rad/s    |
| 8        | PitchDot    | Pitch, angular velocity              | estimated | $\dot{\varTheta}$ | rad/s    |
| 9        | Override    |                                   | output    | $o$               | boolean  |

**License**
The dataset is licensed under the terms of the CC BY-SA 4.0 license.
To view a copy of this license, visit https://creativecommons.org/licenses/by-sa/4.0/

## Source Code

The source code is available in the `src/` folder. The included readme gives a brief introduction on environment setup and on how to run evaluations. If you have further questions please do not hesistate to contact us via email.

**License**
The source code is licensed under the terms of the MIT license.

# Acknowledgement

We would like to thank the authors of the following published code:

https://github.com/decisionintelligence/TFB

https://github.com/Thinklab-SJTU/Crossformer

https://github.com/thuml/Time-Series-Library

# Citation

If you find this dataset or published source code useful, please cite our publication via

```
@inproceedings{SRUH2025,
  title         = {Deep Learning-based Time Series Forecasting for Industrial Discrete Process Data},
  publisher     = {IEEE},
  booktitle     = {{8th IEEE Conference on Industrial Cyber-Physical Systems (ICPS)}},
  address       = {Emden, Germany},
  author        = {Sa\ss{}nick, Olaf and Rosenstatter, Thomas and Unterweger, Andreas and Huber, Stefan},
  month         = may,
  year          = 2025,
  doi           = {10.1109/ICPS65515.2025.11087869},
}
```
