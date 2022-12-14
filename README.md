# mediapipe-based-pose-estimation-interface

This is a user interface written by `PyQt`, based on `mediapipe` human pose estimation. You can learn about mediapipe from [here](https://github.com/google/mediapipe)

This project is realized on **Windows** platform, but should work on **Linux** as well.

## Requirements
The project requires following packages:

|package name| version|
|-----|-----|
|PyQt5         |               5.15.4|
|PyOpenGL        |             3.1.6|
|matplotlib       |            3.6.2|
|mediapipe        |            0.9.0|
|opencv-python     |           4.6.0.66|

The `Camera Selection` module requires drive for DaHeng cameras. You can download the drive from [here](https://www.daheng-imaging.com/downloads/)

To run `main.py`, you have to include `gxipy` folder in your project location.

## Use

Simply run `main.py`

## Current File

|Name| Usage|
|----|----|
|main.py|core code|
|main_ui.py|python ui realize for current file.|
|\*.ui|PyQt UI defination file|
