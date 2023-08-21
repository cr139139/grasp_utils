import numpy as np
import pybullet as p


def get_point_cloud(width, height, view_matrix, proj_matrix):
    image_arr = p.getCameraImage(width=width, height=height,
                                 viewMatrix=view_matrix, projectionMatrix=proj_matrix,
                                 renderer=p.ER_TINY_RENDERER)
    depth = image_arr[3]
    segmentation = image_arr[4]

    # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
    proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
    view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
    tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

    # create a grid with pixel coordinates and depth values
    y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
    y *= -1.
    x, y, z, s = x.reshape(-1), y.reshape(-1), depth.reshape(-1), segmentation.reshape(-1)
    h = np.ones_like(z)

    pixels = np.stack([x, y, z, h], axis=1)
    # filter out "infinite" depths
    pixels = pixels[z < 0.999]
    s = s[z < 0.999]

    pixels = pixels[s == 1]

    pixels[:, 2] = 2 * pixels[:, 2] - 1

    # turn pixels to world coordinates
    points = np.matmul(tran_pix_world, pixels.T).T
    points /= points[:, 3: 4]
    points = points[:, :3]

    return points


def draw_point_cloud(points):
    colors = np.zeros(points.shape)
    colors[:, 0] = 1 - ((points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min()) * 1)
    colors[:, 1] = ((points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min()) * 1)
    colors[:, 2] = 1
    p.addUserDebugPoints(points, colors, pointSize=2.)


def draw_grasp_poses(H, color=[1, 0, 0]):
    gripper_vertices = np.array([[4.10000000e-02, 0, 6.59999996e-02, 1],
                                 [4.10000000e-02, 0, 1.12169998e-01, 1],
                                 [-4.100000e-02, 0, 6.59999996e-02, 1],
                                 [-4.100000e-02, 0, 1.12169998e-01, 1],
                                 [0, 0, 0, 1],
                                 [0, 0, 6.59999996e-02, 1]])
    gripper_edges = np.array([[4, 5], [0, 2], [0, 1], [2, 3]])
    for i in range(H.shape[0]):
        gripper_points = (H[i] @ gripper_vertices.T).T[:, :3]
        for u, v in gripper_edges:
            p.addUserDebugLine(gripper_points[u], gripper_points[v], lineColorRGB=color, lineWidth=2)


