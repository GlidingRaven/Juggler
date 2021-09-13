# Juggler platform

![Real robot and simulation](files/ro.jpg)

The ball begins to fall from a given point and the robot tries to bounce it. The kicking platform can rotate on two axes and move vertically.
This repository includes an accurate physical simulation and a complete solution based on ML approach
 
[Simulation in Colab (no GUI)](https://colab.research.google.com/drive/1CBMK3y_V7m3XDEyIOxfX7YjOGRlooMEp?usp=sharing)

### Model parameters
To run 1 simulation you need some input data:
* Coordinates of ball (x, y, z)
* Velocity of ball (x, y)
* Angles of platform (alpha, beta)
* Vertical velocity of platform (z-vel)
* Response delay (to kick the ball in time)

After execution the program returns a score from 0 to 1 based on the three grades A, B, C.
* A — contact point location. The farther from the middle the worse it is
* B — maximum rebound height. The closer to the target altitude the better
* C — lifting of the platform at the moment of contact. The closer to the middle of the maximum lift the better. Necessary for the stable operation of the real device

[Read more about my math ...](https://rust-donkey-1a4.notion.site/Juggling-robot-RL-solution-a3202e2119df45d9ae70768b8373bae2)

## Requirements
Python IDE, Pybullet and a few more libraries