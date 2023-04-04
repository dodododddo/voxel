import taichi as ti
from taichi.math import *
ti.init(arch=ti.x64)

from voxel import Voxelmap
from read_points import read_points
from scene import Scene
vec3i = ti.types.vector(3,ti.i32)

scene = Scene(voxel_edges=0, exposure=2) # 创建场景，指定体素描边宽度和曝光值
scene.set_floor(0, (1.0, 1.0, 1.0)) # 地面高度
scene.set_background_color((0.5, 0.5, 0.6)) # 天空颜色
scene.set_directional_light((1, 1, -1), 0.2, (1, 0.8, 0.6)) # 光线方向和颜色

@ti.kernel
def initialize_voxels():
    for i,j,k in ti.ndrange(map.size[0],map.size[1],map.size[2]):
       if map.probability_grid[i,j,k] > 0.6:
           scene.set_voxel(vec3(i,j,k),1,vec3(1))

map = Voxelmap()
points = read_points('points.txt')
map.add_points_batch(points)
map.flush_map()
initialize_voxels()
scene.finish()

