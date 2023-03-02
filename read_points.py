import numpy as np
import taichi as ti

ti.init(arch = ti.cpu)
vec3i = ti.types.vector(3,ti.i32)

def read_points(path:str)-> ti.field:
    points_np = np.genfromtxt(path, delimiter=' ')[:,[1,2,0]].astype(np.int32) # z轴换到第三位
    points = ti.field(ti.i32,shape = points_np.shape)
    points.from_numpy(points_np)
    return points
