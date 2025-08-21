import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

# Load model and data
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# PD control gains
Kp = np.array([100, 100, 100, 50, 50, 50])  
Kd = np.array([2, 2, 2, 1, 1, 1])   
Ki = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1]) 
num_joints = model.nu

# Sinusoidal trajectory parameters
duration = 500.0  # simulation time in seconds
frequency = 0.02  # Hz
amplitude = np.deg2rad(30)  # ±30 degrees

# Trajectory history buffers
q_des_history = []
q_act_history = []
time_history = []

# Initial pose
initial_qpos = np.copy(data.qpos[:num_joints])

# Run simulation on main thread
def run_simulation():
    sim_start = data.time
    integral_error = np.zeros(num_joints)
    for _ in range(int(duration / model.opt.timestep)):
            t = data.time - sim_start

            # Desired position & velocity
            q_des = initial_qpos + amplitude * np.sin(2 * np.pi * frequency * t + np.arange(num_joints))
            qd_des = amplitude * 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * t + np.arange(num_joints))

            # Errors
            q_err = q_des - data.qpos[:num_joints]
            qd_err = qd_des - data.qvel[:num_joints]
            integral_error += q_err * model.opt.timestep

            # PD control law - without I control - there might be some steady state control; plotting two (qd vs data.qpos) against 
            # other. also plot the control - see how this is and how FF does against this. 
            # PID/PD + FF vs PD 
            data.ctrl[:num_joints] = Kp * q_err + Kd * qd_err 

            # Log data
            q_des_history.append(q_des.copy())
            q_act_history.append(data.qpos[:num_joints].copy())
            time_history.append(t)

            # Step sim
            mujoco.mj_step(model, data)

if __name__ == "__main__":
    run_simulation()

    # After simulation: convert lists to arrays
    q_des_history = np.array(q_des_history)
    q_act_history = np.array(q_act_history)
    time_history = np.array(time_history)

    # Plot results
    plt.figure(figsize=(12, 8))
    for i in range(num_joints):
        plt.subplot(num_joints, 1, i + 1)
        plt.plot(time_history, q_des_history[:, i], label=f'Joint {i+1} Desired')
        plt.plot(time_history, q_act_history[:, i], label=f'Joint {i+1} Actual', linestyle='--')
        plt.ylabel("Position (rad)")
        plt.legend()
        if i == 0:
            plt.title("Joint Tracking: Desired vs Actual")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()
