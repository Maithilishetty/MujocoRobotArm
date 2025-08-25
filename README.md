## Mujoco Robot Arm

Experiments to learn Mujoco and setting up traditional + RL controllers for a robotic arm environment within in. 

To install all the required libraries, run,  
`pip install -r requirements.txt`

Next, run the example,  
`python3 main_IK.py`

If you are on MacOS, you will likely have to use `mjpython` instead of `python3` to run the example. See [here](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer) for details. 

Once run, you'll see a mujoco simulation and 3 figures stored in your folder namely, 
- `joint_tracking_plot_pos.png` : Plot for Joint Position Tracking
- `joint_tracking_plot_vel.png` : Plot for Joint Velocity Tracking
- `tau_plot.png` : Plot for Torque Inputs
- `cartesian_tracking_plot.png` : Plot for Cartesian Position Tracking 