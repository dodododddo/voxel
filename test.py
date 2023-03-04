from voxel import Voxelmap,vec3i
from read_points import read_points
import taichi as ti
ti.init(arch = ti.cpu)



testpoints = read_points('points.txt')
print(testpoints[78997,2])
print(testpoints.shape)

testmap = Voxelmap()
testmap.add_points_batch(testpoints)
visit1 = testmap.visit_grid(vec3i(900,1000,1000))
visit2 = testmap.visit_grid(vec3i(963,919,866))
visit3 = testmap.visit_grid(vec3i(1000,1000,0))
print('visit1 = {},visit2 = {},visit3 = {}'.format(visit1,visit2,visit3))

testmap.flush_map()
visit1 = testmap.visit_grid(vec3i(900,1000,1000))
visit2 = testmap.visit_grid(vec3i(963,919,866))
visit3 = testmap.visit_grid(vec3i(1000,1000,2000))
print('visit1 = {},visit2 = {},visit3 = {}'.format(visit1,visit2,visit3))