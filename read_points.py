import numpy as np
import taichi as ti

ti.init(arch = ti.x64)
vec3i = ti.types.vector(3,ti.i32)

def read_points(path:str)-> ti.field:
    points_np = np.genfromtxt(path, delimiter=' ')[:,[1,2,0]].astype(np.int32) # z轴换到第三位
    points = ti.field(ti.i32,shape = points_np.shape)
    points.from_numpy(points_np)
    return points

def read_points_fake(path:str)->ti.field: # 硬造的点云
    points_np = np.genfromtxt(path, delimiter=' ')[:,[1,2,0]].astype(np.int32) # z轴换到第三位
    points_np[:,1] *= -1
    points_np //= 3
    points = ti.field(ti.i32,shape = points_np.shape)
    points.from_numpy(points_np)
    return points
    