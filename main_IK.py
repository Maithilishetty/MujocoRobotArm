from Gauss_Newton_IK import GaussNewtonIK
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

# Load model and data
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
data_copy = mujoco.MjData(model)

# PI control gains
Kp = np.array([100, 10, 10, 50, 50, 50])  
Ki = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1]) 

num_joints = model.nu

# Sinusoidal trajectory parameters
duration = 10.0  # simulation time in seconds
frequency = 0.02  # Hz
amplitude = np.deg2rad(30)  # ±30 degrees

# Trajectory history buffers
q_des_history = []
qd_des_history = []
q_act_history = []
qd_act_history = []
tau_history = []
time_history = []

# Initial pose
initial_qpos = np.copy(data.qpos[:num_joints])
q_des = initial_qpos; 

# Initialize IK solver
ik_solver = GaussNewtonIK(model, data)

# Integral term
integral_error = np.zeros(num_joints)
sim_start = data.time
base_target = np.array([0.25, 0.5, 0.5])

# Sinusoid tracking 
amp_x = 0.1 
amp_y = 0.1 
amp_z = 0.1 
freq_x = 0.1
freq_y = 0.2 
freq_z = 0.1

body_name = "wrist_3_link"
body_id = model.body(body_name).id
jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))

with mujoco.viewer.launch_passive(model, data) as viewer:
    for _ in range(int(duration / model.opt.timestep)):
        t = data.time - sim_start
        
        J_pos = np.zeros(3 * model.nv)
        J_rot = np.zeros(3 * model.nv)

        target_pos = base_target + np.array([amp_x * np.sin(2 * np.pi * freq_x * t),amp_y * np.sin(2 * np.pi * freq_y * t),amp_z * np.sin(2 * np.pi * freq_z * t)])
        
        # uncomment to use Inverse Kinematics using Gauss Newton Solver 
        # q_des = ik_solver.solve(target_pos, body_name="wrist_3_link", q_init=data.qpos[:model.nq])
        
        x_vel = np.array([amp_x*(2*np.pi*freq_x)*np.cos(2 * np.pi * freq_x * t), amp_y*(2*np.pi*freq_y)*np.cos(2 * np.pi * freq_y * t), amp_z*(2*np.pi*freq_z)*np.cos(2 * np.pi * freq_z * t)])
        x_acc = np.array([-amp_x *(2*np.pi*freq_x)**2*np.sin(2 * np.pi * freq_x * t),-amp_y*(2*np.pi*freq_y)**2 * np.sin(2 * np.pi * freq_y * t), -amp_z*(2*np.pi*freq_z)**2 * np.sin(2 * np.pi * freq_z * t)])
        
        x_current = data.body(body_id).xpos.copy()
        
        mujoco.mj_jac(model, data, jacp, jacr, x_current, body_id)  
        J = jacp[:, :model.nv]
        
        qd_des = np.linalg.pinv(J) @ x_vel  
        q_des = q_des + qd_des*model.opt.timestep #integrating qd_des to get q_des 
        qdd_des = np.linalg.pinv(J) @ x_acc
        
        data.qpos[:num_joints] = q_des
        data.qvel[:num_joints] = qd_des
        data.qacc[:num_joints] = qdd_des  
        
        mujoco.mj_inverse(model, data)  
        tau = data.qfrc_inverse[:num_joints].copy()    
        
        # noise terms; not using this rn (can change this to disturbance or parameter uncertainty to make feedback control more interesting)
        # noise_q = np.random.normal(0, 0.005, size=num_joints)  
        # noise_qd = np.random.normal(0, 0.01, size=num_joints) 

        q_meas = data.qpos[:num_joints] 
        qd_meas = data.qvel[:num_joints] 
        q_err = q_des - q_meas
        qd_err = qd_des - qd_meas
        integral_error += q_err * model.opt.timestep

        # Feedforward + PI controller
        data.ctrl[:num_joints] = tau + Kp * q_err + Ki * integral_error

        # Log
        tau_history.append(tau.copy())
        q_des_history.append(q_des.copy())
        qd_des_history.append(qd_des.copy())
        q_act_history.append(q_meas.copy())
        qd_act_history.append(qd_meas.copy())
        
        time_history.append(t)

        mujoco.mj_step(model, data)
        viewer.sync()  

# After simulation: convert lists to arrays
tau_history = np.array(tau_history)
rows, columns = tau_history.shape
print(f"Matrix dimensions: {rows} rows, {columns} columns")
q_des_history = np.array(q_des_history)
qd_des_history = np.array(qd_des_history)
q_act_history = np.array(q_act_history)
qd_act_history = np.array(qd_act_history)
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

plt.savefig("joint_tracking_plot_pos.png")

# Plot results
plt.figure(figsize=(12, 8))
for i in range(num_joints):
    plt.subplot(num_joints, 1, i + 1)
    plt.plot(time_history, qd_des_history[:, i], label=f'Joint {i+1} Desired')
    plt.plot(time_history, qd_act_history[:, i], label=f'Joint {i+1} Actual', linestyle='--')
    plt.ylabel("Velocity (rad/s)")
    plt.legend()
    if i == 0:
        plt.title("Joint Tracking: Desired vs Actual")
plt.xlabel("Time (s)")
plt.tight_layout()

plt.savefig("joint_tracking_plot_vel.png")

plt.figure(figsize=(12, 8))
for i in range(num_joints):
    plt.subplot(num_joints, 1, i + 1)
    plt.plot(time_history, tau_history[:, i], label=f'Joint {i+1}')
    plt.ylabel("Torque")
    plt.legend()
plt.xlabel("Time (s)")
plt.tight_layout()

plt.savefig("tau_plot_vel.png")
