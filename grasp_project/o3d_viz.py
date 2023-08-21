import numpy as np
import open3d as o3d
from gpis import RGPIS

mesh = o3d.io.read_triangle_mesh("../006_mustard_bottle/poisson/textured.obj", True)
mesh.compute_vertex_normals()
# theta = np.pi/4
# mesh.transform([[np.cos(theta), -np.sin(theta), 0, 0],
#                 [np.sin(theta), np.cos(theta), 0, 0],
#                 [0, 0, 1, 0],
#                 [0, 0, 0, 1]])

pcd = mesh.sample_points_poisson_disk(number_of_points=5000, init_factor=5)

diameter = 1
camera = [0, diameter, 0]
radius = diameter * 100
_, pt_map = pcd.hidden_point_removal(camera, radius)
pcd = pcd.select_by_index(pt_map)
points_temp = np.asarray(pcd.points)
scales = (points_temp[:, 1] - np.min(points_temp[:, 1])) / (np.max(points_temp[:, 1]) - np.min(points_temp[:, 1]))
colors = np.ones(np.asarray(pcd.points).shape) * 0.
# colors[:, 0] = 0.75 * (1 - scales)
colors[:, 1] = scales
colors[:, 2] = 1 - scales
pcd.colors = o3d.utility.Vector3dVector(colors)

gp = RGPIS()
gp.train(np.asarray(pcd.points))
points, normals = gp.get_surface_points(samples=16)
var = gp.get_var(points)
var = np.diag(var)
colors = np.ones(points.shape) * 0.
colors[:, 0] = (var - np.min(var)) / (np.max(var) - np.min(var))
colors[:, 2] = 1 - (var - np.min(var)) / (np.max(var) - np.min(var))

pcd_recon = o3d.geometry.PointCloud()
pcd_recon.points = o3d.utility.Vector3dVector(points)
pcd_recon.normals = o3d.utility.Vector3dVector(-normals)
pcd_recon.colors = o3d.utility.Vector3dVector(colors)

import copy

mesh_rot = copy.deepcopy(mesh)
pcd_rot = copy.deepcopy(pcd)
pcd_recon_rot = copy.deepcopy(pcd_recon)

pcd.transform([[1, 0, 0, -0.15],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])
pcd_recon.transform([[1, 0, 0, -0.3],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

mesh_rot.transform([[-1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, 1, -0.2],
                    [0, 0, 0, 1]])
pcd_rot.transform([[-1, 0, 0, -0.15],
                   [0, -1, 0, 0],
                   [0, 0, 1, -0.2],
                   [0, 0, 0, 1]])
pcd_recon_rot.transform([[-1, 0, 0, -0.3],
                         [0, -1, 0, 0],
                         [0, 0, 1, -0.2],
                         [0, 0, 0, 1]])

o3d.visualization.draw_geometries([mesh, pcd, pcd_recon, mesh_rot, pcd_rot, pcd_recon_rot],
                                  zoom=0.9,
                                  front=[np.sin(0.5), np.cos(0.5), 0],
                                  lookat=[-0.15, 0, 0],
                                  up=[0, 0, 1]
                                  )
