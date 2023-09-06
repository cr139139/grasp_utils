import copy

import numpy as np
import pybullet as p
import pybullet_data as pd
import time
import open3d as o3d

from pybullet_utils import get_point_cloud, draw_point_cloud, draw_grasp_poses, draw_grasp_frames
from grasp_sampler import GraspSampler
from riem_gmm import SE3GMM
from robot import KUKASAKE

import torch
import pytorch_kinematics as pk
import pytransform3d.trajectories

model = GraspSampler()

p.connect(p.GUI)
dt = 1. / 240.
SOLVER_STEPS = 1000  # a bit more than default helps in contact-rich tasks
TIME_SLEEP = dt * 3  # for visualization
LATERAL_FRICTION = 1.0000
SPINNING_FRICTION = 0.0001
ROLLING_FRICTION = 0.0001
p.setPhysicsEngineParameter(fixedTimeStep=dt, numSolverIterations=SOLVER_STEPS,
                            useSplitImpulse=True, enableConeFriction=True,
                            splitImpulsePenetrationThreshold=0.0)
p.resetSimulation()

p.configureDebugVisualizer(rgbBackground=[0, 0, 0])
p.setAdditionalSearchPath(pd.getDataPath())
p.resetDebugVisualizerCamera(1, 0, 0, [0, 0, 0])

plane_id = p.loadURDF('plane.urdf', basePosition=[0., 0., -0.626], useFixedBase=True)
# object_id = p.loadURDF('duck_vhacd.urdf', basePosition=[0.65, 0.0, 0.3], globalScaling=0.7)
# object_id = p.loadURDF('../006_mustard_bottle/tsdf/textured.urdf', basePosition=[0.6, 0.0, 0.0],
#                        baseOrientation=[0, 0, 0, 1], globalScaling=8, useFixedBase=False)
filename = '../../ycb_dataset_mesh/ycb/014_lemon/google_16k/textured_convex.obj'

import trimesh
mesh = trimesh.load_mesh(filename)

bounds = np.array(mesh.bounds)
bounds = np.min(bounds[1] - bounds[0])
if bounds > 0.04:
    scale = 0.04 / bounds
else:
    scale = 1

center_mass = np.array(mesh.center_mass) * scale

collisionShapeId_VHACD = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                fileName=filename, meshScale=[scale, scale, scale])
object_id = p.createMultiBody(baseMass=0.1,
                              baseCollisionShapeIndex=collisionShapeId_VHACD,
                              basePosition=[0.65, 0.0, 0.1],
                              baseInertialFramePosition=center_mass)

# table_id = p.loadURDF('/table/table.urdf', basePosition=[0.5, 0., -0.626], globalScaling=1., useFixedBase=True)
# p.setGravity(0, 0, -9.81)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

robot = KUKASAKE([0, 0, 0], [0, 0, 0, 1])
robot.reset_arm_poses(np.deg2rad([0, 30, 0, -60, 0, 90, 0]))
robot.open_gripper()
robot.control_arm_poses(np.deg2rad([0, 30, 0, -60, 0, 90, 0]))


p.changeDynamics(object_id, -1, lateralFriction=LATERAL_FRICTION, spinningFriction=SPINNING_FRICTION,
                 rollingFriction=ROLLING_FRICTION)
for i in range(p.getNumJoints(robot.robot_id)):
    p.changeDynamics(robot.robot_id, i, lateralFriction=LATERAL_FRICTION, spinningFriction=SPINNING_FRICTION,
                     rollingFriction=ROLLING_FRICTION)

for i in range(1000):
    p.stepSimulation()
    # time.sleep(0.01)

points = get_point_cloud(640, 480, robot.get_view_matrix(), robot.projectionMatrix, object_id)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd = pcd.voxel_down_sample(voxel_size=0.01)
points = np.asarray(pcd.points)
points, H, w = model.sampler(points)

gmm_reach = SE3GMM()
parameter = np.load('gmm_reachability.npz')
gmm_reach.mu = parameter['mu']
gmm_reach.sigma = parameter['sigma']
gmm_reach.pi = parameter['pi']
H_prob = gmm_reach.eval(H)

H = H[H_prob > 0.0005733336724437366]
H = H[H[:, 2, 3] > 0.05]
# draw_grasp_poses(H, color=[0.6, 0.6, 0.6], robot='kuka')
draw_point_cloud(points)

gmm_grasp = SE3GMM()
H_sample = gmm_grasp.fit(H, n_clusters=min(4, H.shape[0]), n_iterations=10)
H_mu = gmm_grasp.mu

prob = gmm_grasp.eval(H)
draw_grasp_poses(H_mu, color=[1, 0., 0.], robot='kuka')
# T = pytransform3d.trajectories.exponential_coordinates_from_transforms(robot.get_ee_transform())
# for i in range(100):
#     grad = gmm_grasp.grad(T)
#     norm = np.linalg.norm(grad)
#     if norm == 0:
#         norm = 1
#     if i % 5 == 0:
#         draw_grasp_poses(pytransform3d.trajectories.transforms_from_exponential_coordinates(T[np.newaxis, :]),
#                          color=[0, 1, 0], robot='kuka')
#     if np.linalg.norm(grad) < 0.6:
#         break
#     print(np.linalg.norm(grad))
#     if np.linalg.norm(grad) > 1e2:
#         grad = grad / np.linalg.norm(grad) * 1e2
#     T += 1e-4 * grad
#     # T[:3] += 1e-3 * grad[:3] / np.linalg.norm(grad[:3])
#     # T[3:] += 1e-3 * grad[3:] / np.linalg.norm(grad[3:])

device = "cpu"
dtype = torch.float32
chain = pk.build_serial_chain_from_urdf(open("KUKA_IIWA_URDF/iiwa7.urdf").read(), "iiwa_link_ee")
chain = chain.to(dtype=dtype, device=device)
q_orig = robot.get_joint_state()
count = 500
flag = False

q_temp = torch.tensor(robot.get_joint_state(), dtype=dtype, device=device, requires_grad=False)[None, :]
with torch.inference_mode():
    while True:

        for i in range(1):
            m = chain.forward_kinematics(q_temp, end_only=True).get_matrix()
            T = pytransform3d.trajectories.exponential_coordinates_from_transforms(m[0].numpy())

            grad = gmm_grasp.grad(T)
            if np.linalg.norm(grad) > 1e1:
                grad = grad / np.linalg.norm(grad) * 1e1
            T_new = copy.deepcopy(T)
            T_new[:3] = T[:3] + 1e-3 * grad[:3]
            T_new[3:] = T[3:] + 1e-3 * grad[3:]
            T_new = torch.tensor(pytransform3d.trajectories.transforms_from_exponential_coordinates(T_new),
                                 dtype=dtype)[None, :, :]

            pos = T_new[:, :3, 3] - m[:, :3, 3]
            rot = pk.matrix_to_axis_angle(T_new[:, :3, :3] @ m[:, :3, :3].transpose(1, 2))
            ee_e = torch.cat([pos, rot], dim=1)

            J = chain.jacobian(q_temp)
            q_temp += (J.transpose(1, 2) @ torch.linalg.solve(J @ J.transpose(1, 2),
                                                              ee_e[:, :, None]))[:, :, 0]

        print(torch.linalg.norm((J.transpose(1, 2) @ torch.linalg.solve(J @ J.transpose(1, 2), ee_e[:, :, None]))[:, :, 0]))
        if torch.linalg.norm((J.transpose(1, 2) @ torch.linalg.solve(J @ J.transpose(1, 2), ee_e[:, :, None]))[:, :, 0]) < 0.0005:
            break

        # if (torch.linalg.norm(
        #         q_temp - torch.tensor(robot.get_joint_state(), dtype=dtype, device=device, requires_grad=False)[None,
        #                  :])) < 0.003:
        #     robot.close_gripper()
        #     flag = True
        # if flag:
        #     count -= 1
        # if count == 0:
        #     break
        robot.control_arm_poses(q_temp[0])
        robot.reset_arm_poses(q_temp[0])
        # p.stepSimulation()
        # time.sleep(0.01)
robot.reset_arm_poses(q_temp[0])
robot.control_arm_poses(q_temp[0])
for i in range(240 * 2):
    robot.control_arm_poses(q_temp[0])
    robot.close_gripper()
    p.stepSimulation()

p.setGravity(0, 0, -9.81)
for i in range(240 * 10):
    q_curr = robot.get_joint_state()
    robot.control_arm_poses(0.01 * (q_orig-q_curr) + q_curr)
    p.stepSimulation()

p.disconnect()
