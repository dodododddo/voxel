# 点云体素化
## 目前想法：
1. 用Taichi的SNode写一个八叉树地图,利用Taichi的自动并行和内存访问优化，保证体素化过程速度。
2. 用fast-lio中的点云处理方式
3. 体素化地图可以用来跑Voxelnet。

## 具体实现
Class Voxelmap: 体素地图类，私有变量：八叉树最大层数，分辨率，最大尺寸
实现方法：init,add_point,visit,show,flush

