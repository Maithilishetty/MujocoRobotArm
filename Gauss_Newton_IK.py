import numpy as np
import mujoco

class GaussNewtonIK:
    def __init__(self, model, data, step_size=0.5, tol=1e-3, max_iters=100):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.max_iters = max_iters
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

    def check_joint_limits(self, q):
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0], min(q[i], self.model.jnt_range[i][1]))
        return q

    def solve(self, target_pos, body_name, q_init=None):
        if q_init is None:
            q_init = self.data.qpos[:self.model.nq].copy()

        q = q_init.copy()
        self.data.qpos[:self.model.nq] = q
        mujoco.mj_forward(self.model, self.data)
        body_id = self.model.body(body_name).id
        x_current = self.data.body(body_id).xpos.copy()
        err = target_pos - x_current

        iters = 0
        while np.linalg.norm(err) > self.tol and iters < self.max_iters:
            mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, x_current, body_id)
            J = self.jacp[:, :self.model.nv]
            dq = np.linalg.pinv(J) @ err
            q += self.step_size * dq
            q = self.check_joint_limits(q)
            self.data.qpos[:self.model.nq] = q
            mujoco.mj_forward(self.model, self.data)
            x_current = self.data.body(body_id).xpos.copy()
            err = target_pos - x_current
            iters += 1

        return q
