import taichi as ti
from read_points import read_points
ti.init(arch = ti.cpu,default_fp = ti.f32,default_ip = ti.i32)

vec3f = ti.types.vector(3,ti.f32)
vec3i = ti.types.vector(3,ti.i32)

@ti.data_oriented
class Voxelmap(object):
    def __init__(self,ppi:int = 2,size:vec3i = vec3i(1500,250,2500)): # size默认值为rmuc场地全图尺寸，单位为cm；ppi默认值为2cm
        self.size = size
        self.ppi = ppi

        self.probality_grid = ti.field(ti.f32) # 体素地图的叶结点存放该体素被占据的logit变换后的概率：y = log(x/1-x)
        self.block1 = ti.root.pointer(ti.ijk,(8,4,16)) # 一级方块指针池
        self.block2 = self.block1.pointer(ti.ijk,(8,4,8)) # 二级方块指针池
        self.block3 = self.block2.pointer(ti.ijk,(8,4,8)) # 三级方块指针池
        self.grid = self.block3.bitmasked(ti.ijk,(2,2,2)) # 每个体素的激活状态是独立的
        self.grid.place(self.probality_grid)
        assert(self.probality_grid.shape[0] * self.ppi >= self.size[0] \
               and self.probality_grid.shape[1] * self.ppi >= self.size[1] \
               and self.probality_grid.shape[2] * self.ppi >= self.size[2]) 
        # 确保体素地图能覆盖预设范围
    
    @ti.kernel
    def add_point(self,point:vec3i)-> bool: # 逐点更新
        success_add = self.is_inside(point)  # 判断点是否在地图范围内
        if success_add:
            grid_increase = self.Pos2Grid(point) 
            self.probality_grid[grid_increase] += 0.6  # 该点对应占据体素位置占据概率对数值上升

            regu = ti.max(ti.abs(point[0]),-1 * point[1],point[2]) 
            grad = point / regu      # 找到x,y,z轴下降最缓的方向，保证沿此方向每走一部只经过一个体素
            for i in range(regu):
                grid_reduce = self.Pos2Grid(point - grad * (i + 1)) # 沿此方向下降搜索可能遮挡该点的体素，直到到达原点附近体素
                if ti.is_active(self.grid,grid_reduce):
                    self.probality_grid[grid_reduce] -= 0.05 # 未被激活的体素不考虑，已被激活的体素认为被占据的概率下降
                                                             # 若点到原点之间的体素被占据，本应检测不到该点

        return success_add
                
             
    @ti.kernel
    def add_points_batch(self,points:ti.template()): # 批量点云更新，如逐帧点云更新
        for i in range(points.shape[0]):
            point = vec3i(points[i,0],points[i,1],points[i,2])
            if self.is_inside(point):
                grid_increase = self.Pos2Grid(point) 
                self.probality_grid[grid_increase] += 0.6  # 该点对应占据体素位置占据概率对数值上升

                regu = ti.max(ti.abs(point[0]),-1 * point[1],point[2]) 
                grad = point / regu      # 找到x,y,z轴下降最缓的方向，保证沿此方向每走一部只经过一个体素
                for j in range(regu):
                    grid_reduce = self.Pos2Grid(point - grad * (j + 1)) # 沿此方向下降搜索可能遮挡该点的体素，直到到达原点附近体素
                    if ti.is_active(self.grid,grid_reduce):
                        self.probality_grid[grid_reduce] -= 0.05 # 未被激活的体素不考虑，已被激活的体素认为被占据的概率下降
                                                                 # 若点到原点之间的体素被占据，本应检测不到
        # TODO: 同一帧点云之间不应该考虑遮挡衰减，可以改为先对一帧点云计算遮挡，再对该帧点云计算占据
        # TODO: 第二级for循环是串行的，暂时没查到怎么让两层for循环都并行

    @ti.kernel
    def visit_grid(self,grid:vec3i)-> ti.f32:
        probablity = 0.0
        if ti.is_active(self.grid,[grid[0],grid[1],grid[2]]):
            probablity =  1 - 1 / (1 + ti.math.exp(self.probality_grid[grid[0],grid[1],grid[2]]))
        return probablity
    

    @ti.kernel
    def flush_map(self):# 刷新地图，把太稀疏的一级结点失活。
        for i,j,k in ti.ndrange(8,4,16):
            if ti.is_active(self.block1,[i,j,k]):
                activation = 0
                for x,y,z in ti.ndrange(8,4,8):
                    if ti.is_active(self.block2,[8 * i + x,4 * j + y,8 * k + z]):
                        activation += 1 
                if activation < 32:
                    ti.deactivate(self.block1,[i,j,k])


    @ti.func
    def Pos2Grid(self,point:vec3i)->vec3i: 
        # 假设雷达中心位置为己方高台（离地250cm)中心，场地范围为1500cm ✖ 2500cm
        return vec3i((point[0] + self.size[0] / 2) // self.ppi,  (point[1] + self.size[1]) // self.ppi,  point[2] // self.ppi)
    

    @ti.func
    def is_inside(self,point:vec3i)-> bool:
        check = False
        if point[0] <= self.size[0] / 2 and point[0] >= -1 * self.size[0] / 2 \
            and point[1] < 0 and point[1] >= -1 * self.size[1] \
            and point[2] > 0 and point[2] <= self.size[2]:
            check = True
        return check


if __name__ == '__main__':
    voxelmap = Voxelmap()
    points = read_points('points.txt')
    voxelmap.add_points_batch(points)
    voxelmap.flush_map()




    



   
            


        
