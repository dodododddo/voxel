import taichi as ti
from read_points import read_points
ti.init(arch = ti.x64,default_fp = ti.f32,default_ip = ti.i32)
vec3f = ti.types.vector(3,ti.f32)
vec3i = ti.types.vector(3,ti.i32)


@ti.data_oriented
class BEV_map(object):
    def __init__(self):
        self.point = ti.field(vec3i)
        self.zone = ti.root.pointer(ti.ij,(32,32))
        self.pillar = self.zone.pointer(ti.ij,(32,32))
        self.point_list = self.pillar.dynamic(ti.k,32,chunk_size = 16)
        self.point_list.place(self.point)
    
    @ti.kernel
    def add_point(self,point:vec3i):
        pass

    @ti.kernel
    def filter(self,threshold = 3):
        pass
        