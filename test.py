from voxel import Voxelmap,vec3i
from read_points import read_points_fake
import taichi as ti
ti.init(arch = ti.x64)


testpoints = read_points_fake('points.txt')
print(testpoints[78997,0],testpoints[78997,1],testpoints[78997,2])
print(testpoints.shape)

testmap = Voxelmap()
testmap.add_points_batch(testpoints)
visit1 = testmap.visit_grid(vec3i(574,80,395))
visit2 = testmap.visit_grid(vec3i(20,-20,500))
visit3 = testmap.visit_grid(vec3i(0,0,0))
print('visit1 = {},visit2 = {},visit3 = {}'.format(visit1,visit2,visit3))

testmap.flush_map()
visit1 = testmap.visit_grid(vec3i(574,80,395))
visit2 = testmap.visit_grid(vec3i(20,-20,500))
visit3 = testmap.visit_grid(vec3i(0,0,0))
print('visit1 = {},visit2 = {},visit3 = {}'.format(visit1,visit2,visit3))

