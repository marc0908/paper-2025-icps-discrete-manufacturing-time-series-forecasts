# Deep Learning-based Time Series Forecasting for Industrial Discrete Process Data

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

The source code will be available in the `src/` folder in 1-2 weeks.
It is currently undergoing necessary refactoring, as much of the TFB source code (https://github.com/decisionintelligence/TFB) included and modified. 
Our goal is to publish the code so that it only contains own written code, with the TFB source code added as a Git submodule to ensure clear separation between the two."

**License**
The source code is licensed under the terms of the MIT license.


