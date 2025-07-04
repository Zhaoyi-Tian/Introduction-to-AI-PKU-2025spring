import numpy as np
from typing import List
from utils import TreeNode
from simuScene import PlanningMap


### 定义一些你需要的变量和函数 ###
STEP_DISTANCE = 3 #RTT向随机点延伸的距离
TARGET_THREHOLD=0.23 #树上的点距最终点距离小于此值时视为到达终点，结束此树
MAX_N=50
MIN_REACH=0.16 #如果pacman距树上的点小于此值，则视为已经到达此点，切换目标为下一个点，这个值太大了不好
MIN_NEAR=0.15 #若随机点距树最小距离小于此值，则不采纳
'''以上是可能重要的参数'''

RAO=0.5 #微小扰动
STUCK_THRESHOLD=15
RAO_L=[(RAO,0),(-RAO,0),(0,RAO),(0,-RAO)]
### 定义一些你需要的变量和函数 ###

class RRT:
    def __init__(self, walls) -> None:
        """
        输入包括地图信息，你需要按顺序吃掉的一列事物位置 
        注意：只有按顺序吃掉上一个食物之后才能吃下一个食物，在吃掉上一个食物之前Pacman经过之后的食物也不会被吃掉
        """
        self.map = PlanningMap(walls)
        self.walls = walls
        
        # 其他需要的变量
        ### 你的代码 ###      
        self.xmin = np.min(walls, axis=0)[0]
        self.ymin = np.min(walls, axis=0)[1]
        self.xmax = np.max(walls, axis=0)[0]
        self.ymax = np.max(walls, axis=0)[1]
        ### 你的代码 ###
        
        # 如有必要，此行可删除
        self.path = None
        self.idx=0
        self.n=0
        self.stuck_count = 0  # 卡住计数器
        self.replan_attempt = 0  # 重新规划尝试次数
        
    def find_path(self, current_position, next_food):
        """
        在程序初始化时，以及每当 pacman 吃到一个食物时，主程序会调用此函数
        current_position: pacman 当前的仿真位置
        next_food: 下一个食物的位置
        
        本函数的默认实现是调用 build_tree，并记录生成的 path 信息。你可以在此函数增加其他需要的功能
        """
        
        ### 你的代码 ###      

        ### 你的代码 ###
        # 如有必要，此行可删除
        self.path = self.build_tree(current_position, next_food)
        self.idx=0
        self.n=0
        self.k=0
        self.stuck_count = 0  # 卡住计数器
        self.replan_attempt = 0  # 重新规划尝试次数
        
    def get_target(self, current_position, current_velocity):
        """
        主程序将在每个仿真步内调用此函数，并用返回的位置计算 PD 控制力来驱动 pacman 移动
        current_position: pacman 当前的仿真位置
        current_velocity: pacman 当前的仿真速度
        一种可能的实现策略是，仅作为参考：
        （1）记录该函数的调用次数
        （2）假设当前 path 中每个节点需要作为目标 n 次
        （3）记录目前已经执行到当前 path 的第几个节点，以及它的执行次数，如果超过 n，则将目标改为下一节点
        
        你也可以考虑根据当前位置与 path 中节点位置的距离来决定如何选择 target
        
        同时需要注意，仿真中的 pacman 并不能准确到达 path 中的节点。你可能需要考虑在什么情况下重新规划 path
        """
        target_pose = np.zeros_like(current_position)
        ### 你的代码 ###
        if np.linalg.norm(current_velocity) < 0.01:
            self.stuck_count += 1
        else:
            self.stuck_count = 0  # 重置卡住计数器

            # 如果卡住超过阈值，重新规划路径
        if self.stuck_count > STUCK_THRESHOLD:
            self.replan_attempt += 1
            if self.replan_attempt <= 3:  # 最多尝试3次重新规划
                self.find_path(current_position, self.path[-1])
                return self.path[0]  # 返回新路径的第一个点
            else:
                # 尝试推开墙角
                self.replan_attempt=0
                push_distance = min(0.5, 0.1 * self.replan_attempt)
                angle = np.random.uniform(0, 2 * np.pi)
                push_vector = np.array([np.cos(angle), np.sin(angle)]) * push_distance
                return current_position + push_vector


        if self.idx<len(self.path)-1:
            d=(current_position[0]-self.path[self.idx][0])**2+(current_position[1]-self.path[self.idx][1])**2
            if d<MIN_REACH**2:
                self.idx+=1
                self.n=0
                if self.idx<len(self.path):
                    target_pose=self.path[self.idx]
                else:
                    target_pose=self.path[-1]
            elif self.n<MAX_N:
                target_pose=self.path[self.idx]
                self.n+=1
            else:
                self.idx+=1
                if self.idx < len(self.path):
                    target_pose = self.path[self.idx]
                else:
                    target_pose = self.path[-1]
            if current_velocity[0] < 1e-3 and current_velocity[1] < 1e-3\
                    and self.checkround(current_position.tolist(), [float(target_pose[0]), float(target_pose[1])]):
                '''因为特殊原因导致进行过程中无法抵达之前规划的路径，重新规划路径'''
                # if self.k<4:
                #     x=RAO_L[self.k][0]
                #     y=RAO_L[self.k][1]
                #     self.k+=1
                #     self.find_path(current_position, self.path[-1])
                #     return [current_position[0]+x, current_position[1]+y]
                # else:
                self.find_path(current_position, self.path[-1])
                target_pose = self.path[self.idx]
                self.k=0
            else:
                self.k=0
        elif self.n<MAX_N:
            target_pose=self.path[-1]
        else:
            self.find_path(current_position, self.path[-1])
            target_pose = self.path[self.idx]
        ### 你的代码 ###
        return target_pose-0.1*current_velocity
        
    ### 以下是RRT中一些可能用到的函数框架，全部可以修改，当然你也可以自己实现 ###
    def build_tree(self, start, goal):
        """
        实现你的快速探索搜索树，输入为当前目标食物的编号，规划从 start 位置食物到 goal 位置的路径
        返回一个包含坐标的列表，为这条路径上的pd targets
        你可以调用find_nearest_point和connect_a_to_b两个函数
        另外self.map的checkoccupy和checkline也可能会需要，可以参考simuScene.py中的PlanningMap类查看它们的用法
        """
        path = []
        graph: List[TreeNode] = []
        graph.append(TreeNode(-1, start[0], start[1]))
        ### 你的代码 ###
        while True:
            if np.random.rand() > 0.8:  # 30%概率直接采样到目标区域
                x= goal[0]
                y= goal[1]
            else:
                x=np.random.uniform(self.xmin, self.xmax)
                y=np.random.uniform(self.ymin, self.ymax)
            if not self.map.checkoccupy([x,y]):
                nearest_idx, nearest_distance=RRT.find_nearest_point([x,y], graph)
                if nearest_distance<MIN_NEAR:
                    continue
                if nearest_distance>STEP_DISTANCE:
                    bx=graph[nearest_idx].pos[0]
                    by=graph[nearest_idx].pos[1]
                    nx=STEP_DISTANCE*(x-bx)/nearest_distance+bx
                    ny=STEP_DISTANCE*(y-by)/nearest_distance+by
                    if not self.map.checkoccupy([nx,ny]):
                        if not self.checkround([nx,ny],[bx,by]):
                            graph.append(TreeNode(nearest_idx,nx, ny))
                            if (nx-goal[0])**2+(ny-goal[1])**2<TARGET_THREHOLD**2:
                                break
                else:
                    bx = graph[nearest_idx].pos[0]
                    by = graph[nearest_idx].pos[1]
                    if not self.checkround([float(x),float(y)],[bx,by]):
                        graph.append(TreeNode(nearest_idx,x, y))
                        if (x-goal[0])**2+(y-goal[1])**2<TARGET_THREHOLD**2:
                            break
        path.append(np.array([goal[0],goal[1]]))
        path.append(graph[-1].pos)
        parent_idx=graph[-1].parent_idx
        while parent_idx!=-1:
            path.append(graph[parent_idx].pos)
            parent_idx=graph[parent_idx].parent_idx
        path.reverse()

        return self.optimize_path(path)


    @staticmethod
    def find_nearest_point(point, graph):
        """
        找到图中离目标位置最近的节点，返回该节点的编号和到目标位置距离、
        输入：
        point：维度为(2,)的np.array, 目标位置
        graph: List[TreeNode]节点列表
        输出：
        nearest_idx, nearest_distance 离目标位置距离最近节点的编号和距离
        """
        nearest_idx = -1
        nearest_distance = 10000000.
        ### 你的代码 ###
        for i in range(len(graph)):
            l = (graph[i].pos[0] - point[0]) ** 2 + (graph[i].pos[1] - point[1]) ** 2
            if l < nearest_distance:
                nearest_idx = i
                nearest_distance = l
        ### 你的代码 ###
        return nearest_idx, nearest_distance**0.5

    # def connect_a_to_b(self, point_a, point_b):
    #     """
    #     以A点为起点，沿着A到B的方向前进STEP_DISTANCE的距离，并用self.map.checkline函数检查这段路程是否可以通过
    #     输入：
    #     point_a, point_b: 维度为(2,)的np.array，A点和B点位置，注意是从A向B方向前进
    #     输出：
    #     is_empty: bool，True表示从A出发前进STEP_DISTANCE这段距离上没有障碍物
    #     newpoint: 从A点出发向B点方向前进STEP_DISTANCE距离后的新位置，如果is_empty为真，之后的代码需要把这个新位置添加到图中
    #     """
    #     is_empty = False
    #     newpoint = np.zeros(2)
    #     ### 你的代码 ###
    #     point_c = (STEP_DISTANCE * (point_b - point_a)) / distance(point_a, point_b) + point_a
    #     point_A = point_a.tolist()
    #     point_C = point_c.tolist()
    #     if not self.map.checkline(point_A, point_C)[0] and not self.map.checkoccupy(point_C):
    #         # 如果没有障碍物
    #         is_empty = True
    #     else:
    #         is_empty = False
    #     newpoint = point_c
    #     ### 你的代码 ###
    #     return is_empty, newpoint

    def optimize_path(self, path):
        """优化路径，删除不必要的中间点"""
        if len(path) <= 2:
            return path

        optimized = [path[0]]
        current_index = 0

        while current_index < len(path) - 1:
            #总是优先尝试直接连接终点
            p1 = optimized[-1]
            p2 = path[-1]
            if not self.checkround(p1.tolist(), p2.tolist()):
                optimized.append(p2)
                break

            # 从后向前找第一个无碰撞的连接点
            next_index = len(path) - 2  # 跳过终点
            found = False
            while next_index > current_index:
                p1 = optimized[-1]
                p2 = path[next_index]
                if not self.checkround(p1.tolist(), p2.tolist()):
                    optimized.append(p2)
                    current_index = next_index
                    found = True
                    break
                next_index -= 1

            # 没找到合适点则取下一个点
            if not found:
                optimized.append(path[current_index + 1])
                current_index += 1

        # 确保终点被包含
        if not np.array_equal(optimized[-1], path[-1]):
            optimized.append(path[-1])
        return optimized
    def checkround(self,point_A, point_B):
        # r=0.20
        # x1=point_A[0]
        # y1=point_A[1]
        # x2=point_B[0]
        # y2=point_B[1]
        # d=((x1-x2)**2+(y1-y2)**2)**0.5
        # s=(y2-y1)/d
        # c=(x2-x1)/d
        # k1=self.map.checkline([x1-r*s,y1+r*c],[x2-r*s,y2+r*c])[0]
        # k2 = self.map.checkline([x1+r* s, y1-r*c], [x2 + r* s, y2 -r*c])[0]
        # if k1==False and k2==False:
        #     return False
        # else:
        #     return True
        return self.map.checkline(point_A, point_B)[0]