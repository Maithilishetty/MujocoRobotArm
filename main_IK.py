from Gauss_Newton_IK import GaussNewtonIK
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib
from scipy import integrate
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

# Load model and data
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# PI control gains
Kp = np.array([1, 1, 1, 5, 5, 5])  
Kd = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
Ki = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1]) 

num_joints = model.nu

duration = 10.0  # simulation time in seconds

# Trajectory history buffers
q_des_history = []
qd_des_history = []
q_act_history = []
qd_act_history = []
tau_history = []
tau_history_fb = []
time_history = []

# Initial pose
initial_qpos = np.copy(data.qpos[:num_joints])
base_target = np.array([0.25, 0.5, 0.5])
q_des = np.zeros(num_joints)

# Initialize IK solver
ik_solver = GaussNewtonIK(model, data)

# Integral term
integral_error = np.zeros(num_joints)
sim_start = data.time

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
const = 2*np.pi

x_current = data.body(body_id).xpos.copy()

# offset from the ground 
x_base = x_current.copy()
x_base = x_base + base_target

# for cartesian position tracking 
x_des_history = []
x_act_history = []

# damping factor 
lam = 1e-3

# initialize q_des from IK at base pose
q_des = ik_solver.solve(x_base, body_name="wrist_3_link", q_init=data.qpos[:model.nq])
data.qpos[:num_joints] = q_des.copy()
data.qvel[:num_joints] = 0


with mujoco.viewer.launch_passive(model, data) as viewer:
    for _ in range(int(duration / model.opt.timestep)):
        t = data.time - sim_start
    
        target_pos = base_target + np.array([amp_x * np.sin(const * freq_x * t),amp_y * np.sin(const * freq_y * t),amp_z * np.sin(const * freq_z * t)])
        
        #q_des = ik_solver.solve(target_pos, body_name="wrist_3_link", q_init=data.qpos[:model.nq])
        
        # x_vel becomes the target velocity 
        x_vel = np.array([amp_x * (const*freq_x)*np.cos(const * freq_x * t), amp_y*(const * freq_y)*np.cos(const * freq_y * t), amp_z*(const * freq_z)*np.cos(const * freq_z * t)])
        
        # differentiate x_vel to get x_acc 
        x_acc = np.array([-amp_x *(const * freq_x)**2 * np.sin(const * freq_x * t),-amp_y * (const * freq_y)**2 * np.sin(const * freq_y * t), -amp_z*(const * freq_z)**2 * np.sin(const * freq_z * t)])
        
        x_current = data.body(body_id).xpos.copy()
        
        mujoco.mj_jac(model, data, jacp, jacr, x_current, body_id)  
        J = jacp[:, :model.nv]
        m, n = J.shape
        I = np.eye(m)
        
        qd_des = J.T @ np.linalg.inv(J @ J.T + (lam**2) * I) @ x_vel  
        q_des = q_des + qd_des*model.opt.timestep 
        qdd_des = J.T @ np.linalg.inv(J @ J.T + (lam**2) * I) @ x_acc
        
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
        tau_fb = Kp * q_err + Kd*qd_err + Ki * integral_error
        data.ctrl[:num_joints] = tau + tau_fb
        
        # get the cartesian position
        x_act = data.body(body_id).xpos.copy()
        
        # Logging
        x_des_history.append(target_pos.copy())
        x_act_history.append(x_act.copy())
        tau_history.append(data.ctrl[:num_joints].copy())
        q_des_history.append(q_des.copy())
        qd_des_history.append(qd_des.copy())
        q_act_history.append(q_meas.copy())
        qd_act_history.append(qd_meas.copy())
        
        time_history.append(t)

        mujoco.mj_step(model, data)
        viewer.sync()  

# After simulation: convert lists to arrays
tau_history = np.array(tau_history)
q_des_history = np.array(q_des_history)
qd_des_history = np.array(qd_des_history)
q_act_history = np.array(q_act_history)
qd_act_history = np.array(qd_act_history)
time_history = np.array(time_history)
x_des_history = np.array(x_des_history)
x_act_history = np.array(x_act_history)

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

plt.savefig("tau_plot.png")

plt.figure(figsize=(10, 6))
for i, label in enumerate(["X", "Y", "Z"]):
    plt.subplot(3, 1, i+1)
    plt.plot(time_history, x_des_history[:, i], label=f"{label} Desired")
    plt.plot(time_history, x_act_history[:, i], '--', label=f"{label} Actual")
    plt.ylabel("Position (m)")
    plt.legend()
    if i == 0:
        plt.title("End-Effector Cartesian Tracking")
plt.xlabel("Time (s)")
plt.tight_layout()
plt.savefig("cartesian_tracking_plot.png")

