# Juggler platform

![Real robot and simulation](files/ro.jpg)

The ball begins to fall from a given point and the robot tries to bounce it. The kicking platform can rotate on two axes and move vertically.
This repository includes an accurate physical simulation and a complete solution based on ML approach
 
[Simulation in Colab (no GUI)](https://colab.research.google.com/drive/1CBMK3y_V7m3XDEyIOxfX7YjOGRlooMEp?usp=sharing)

## Abstract
It would be wise to divide the entire gaming experience into separate cases. Each case starts with a ball falling from some point and some speed x, y. The robot tries to kick the ball high enough and into the center of the platform. 
Case refers to the ball's coordinates and speed, and the robot's response in the form of the platform's tilt, its speed, and the delay in starting the movement:
* Coordinates of ball (x, y, z)
* Velocity of ball (x, y)
* Angles of platform (alpha, beta)
* Vertical velocity of platform (z-vel)
* Response delay (to kick the ball in time)

To run 1 simulation you need this input data.
After execution the program returns a score from 0 to 1 based on the three grades **A, B, C**.
* A — contact point location. The farther from the middle the worse it is
* B — maximum rebound height. The closer to the target altitude the better
* C — lifting of the platform at the moment of contact. The closer to the middle of the maximum lift the better. Necessary for the stable operation of the real device


![A, B score](files/scoring.jpg)
Graphs of A and B. [Read more about math behind scores...](https://rust-donkey-1a4.notion.site/Juggling-robot-RL-solution-a3202e2119df45d9ae70768b8373bae2)

## Dataset making and preporation
First, we generate random coordinates of the ball and its speed. 
Then we check if the ball can touch the platform. If not, we discard this set, since the platform will not be able to hit it anyway. Good cases are saved in the file **01_checked_dots.csv**

![Real dataset](files/real_data.jpg)






## Requirements
Python IDE, Pybullet and a few more libraries


[Technical details](https://docs.google.com/document/d/1umx8ZsqzESH3lx-r5ZVmqLn8rJtWfMUKcmMkzg1vBDY/edit?usp=sharing)