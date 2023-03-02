import taichi as ti
from read_points import read_points
ti.init(arch = ti.cpu,default_fp = ti.f32,default_ip = ti.i32)

vec3f = ti.types.vector(3,ti.f32)
vec3i = ti.types.vector(3,ti.i32)

@ti.data_oriented
class Voxelmap(object):
    def __init__(self):
        self.probality_grid = ti.field(ti.f32) # 体素地图的叶结点存放该体素被占据的对数概率：y = log(x/1-x)
        self.block1 = ti.root.pointer(ti.ijk,(32,32,32)) # 一级方块指针池
        self.block2 = self.block1.pointer(ti.ijk,(32,32,32)) # 二级方块指针池
        self.grid = self.block2.bitmasked(ti.ijk,(2,2,2))
        self.grid.place(self.probality_grid)
    
    @ti.kernel
    def add_point(self,point:vec3i)-> bool:
        success_add = False
        if point[0] <= 2000 and point[0] >= -2000 \
            and point[1] <= 2000 and point[1] >= -2000\
            and point[2] >= 0 and point[2] <= 4000:
            grid = Pos2Grid(point)
            self.probality_grid[grid[0],grid[1],grid[2]] += 0.5
            success_add = True
        ## TODO：处理遮挡问题
        return success_add
    
    @ti.kernel
    def add_points_batch(self,points:ti.template()):
        for i in range(points.shape[0]):
            if points[i,0] <= 2000 and points[i,0] >= -2000 \
                and points[i,1] <= 2000 and points[i,1] >= -2000\
                and points[i,2] >= 0 and points[i,2] <= 4000:
                grid = Pos2Grid(vec3i(points[i,0],points[i,1],points[i,2]))
                self.probality_grid[grid[0],grid[1],grid[2]] += 0.5
        ## TODO：处理遮挡问题
        
    
    @ti.kernel
    def visit_grid(self,grid:vec3i)-> ti.f32:
        probablity = 0.0
        if ti.is_active(self.grid,[grid[0],grid[1],grid[2]]):
            probablity =  1 - 1 / (1 + ti.math.exp(self.probality_grid[grid[0],grid[1],grid[2]]))
        return probablity
    
    @ti.kernel
    def flush_map(self):
        for i,j,k in ti.ndrange(32,32,32):
            if ti.is_active(self.block1,[i,j,k]):
                activation = 0
                for x,y,z in ti.ndrange(32,32,32):
                    if ti.is_active(self.block2,[32*i + x,32*j + y,32*k + z]):
                        activation += 1
                if activation < 512:
                    ti.deactivate(self.block1,[i,j,k])
@ti.func
def Pos2Grid(point:vec3i)->vec3i:
    return vec3i((point[0] + 2000) // 2,(point[1] + 2000) // 2,point[2] // 2)




largevolmap = Voxelmap()
points = read_points('points.txt')
print(points[78997,2])
print(points.shape)
largevolmap.add_points_batch(points)
print(largevolmap.visit_grid(vec3i(963,919,866)))
largevolmap.flush_map()




    



   
            


        
