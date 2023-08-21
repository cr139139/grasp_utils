import numpy as np
import pybullet as p
import pybullet_data as pd
import time
import open3d as o3d

from grasp_visualization import visualize_grasps
from pybullet_utils import get_point_cloud, draw_point_cloud, draw_grasp_poses
from grasp_sampler import GraspSampler

model = GraspSampler()

p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())
p.loadURDF('plane.urdf')
p.loadURDF('duck_vhacd.urdf', basePosition=[0.25, 0.0, 0.25], globalScaling=1.)
p.setGravity(0, 0, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

width = 720
height = 640
fov = 60
aspect = width / height
near = 0.1
far = 5

view_matrix = p.computeViewMatrix([0, 0, 0.4], [0.4, 0, 0], [0, 0, 1])
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

points = get_point_cloud(width, height, view_matrix, projection_matrix)

import time

start = time.time()

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd = pcd.voxel_down_sample(voxel_size=0.01)
points = np.asarray(pcd.points)
points, H, w = model.sampler(points)

from riem_gmm import SE3GMM

gmm = SE3GMM()
H = gmm.fit(H, n_clusters=3)
H = gmm.mu

print(time.time() - start)
import pytransform3d.trajectories

T = pytransform3d.trajectories.exponential_coordinates_from_transforms(H[0:1])[0]
target = np.zeros(6)
T = np.zeros(6)

for i in range(1000):
    grad = gmm.grad(T)
    norm = np.linalg.norm(grad)
    if norm == 0:
        norm = 1
    diff = (T - target)
    target = T
    if i % 20 == 0:
        draw_grasp_poses(pytransform3d.trajectories.transforms_from_exponential_coordinates(T[np.newaxis, :]),
                         color=[0, 1, 0])
    T += 1e-2 * grad / norm

T = pytransform3d.trajectories.transforms_from_exponential_coordinates(T[np.newaxis, :])

draw_grasp_poses(H)
draw_grasp_poses(T, color=[0, 0, 1])
draw_point_cloud(points)

while True:
    p.stepSimulation()
    time.sleep(0.03)

p.disconnect()