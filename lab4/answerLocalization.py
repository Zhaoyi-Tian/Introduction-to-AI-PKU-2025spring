from typing import List
import numpy as np

from utils import Particle

### 可以在这里写下一些你需要的变量和函数 ###
COLLISION_DISTANCE = 1
MAX_ERROR = 50000
k = 0.6 #计算重采样系数的权重的系数
pos_std = 0.1 #位置扰动的标准差
angle_std = np.pi / 15 #方向扰动的标准差
rate = 1.12 #扩大高权重粒子的比例


### 可以在这里写下一些你需要的变量和函数 ###

def circle_square_collision(circle_center, circle_radius, square_center, square_size):
	"""
    判断圆与正方形是否碰撞

    参数:
    circle_center: 圆的中心点坐标 (x, y)
    circle_radius: 圆半径
    square_center: 正方形中心点坐标 (cx, cy)
    square_size: 正方形边长

    返回:
    True 如果碰撞，否则 False
    """
	# 计算圆心到正方形中心的距离矢量
	dx = circle_center[0] - square_center[0]
	dy = circle_center[1] - square_center[1]

	# 计算圆心在正方形局部坐标系中的位置
	# 局部坐标系以正方形中心为原点，各边平行于坐标轴
	half_size = square_size / 2.0

	# 计算圆心在正方形局部坐标系中到最近边的距离
	# 这个值表示从圆心到正方形边缘的最小距离
	nearest_x = max(0, abs(dx) - half_size)
	nearest_y = max(0, abs(dy) - half_size)

	# 计算实际最短距离
	distance_squared = nearest_x ** 2 + nearest_y ** 2

	# 检查这个最短距离是否小于等于圆的半径
	return distance_squared <= circle_radius ** 2


def generate_uniform_particles(walls, N):
	"""
    在地图的空地上均匀生成指定数量的粒子

    参数:
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息
    N: int, 需要生成的粒子数量

    返回:
    particles: List[Particle], 生成的粒子列表，每个粒子的权重初始化为1/N
    """
	all_particles: List[Particle] = []
	for _ in range(N):
		all_particles.append(Particle(1.0, 1.0, 1.0, 0.0))

	# 计算地图边界
	xmin = np.min(walls, axis=0)[0]
	ymin = np.min(walls, axis=0)[1]
	xmax = np.max(walls, axis=0)[0]
	ymax = np.max(walls, axis=0)[1]

	# 生成不与墙壁碰撞的随机粒子
	i = 0
	while i < N:
		x = np.random.uniform(xmin, xmax)
		y = np.random.uniform(ymin, ymax)
		theta = np.random.uniform(0, 2 * np.pi)
		tag = True

		# 检查粒子是否与任何墙壁发生碰撞
		for wall in walls:
			wx = wall[0]
			wy = wall[1]
			if circle_square_collision([x, y], 0.25, [wx, wy], 1):
				tag = False

		if tag:
			all_particles[i] = Particle(x, y, theta, 1 / N)
			i += 1

	return all_particles


def calculate_particle_weight(estimated, gt):
	"""
    基于估计的距离传感器数据与真实数据的差异计算粒子权重

    参数:
    estimated: np.array, 粒子的距离传感器估计数据
    gt: np.array, Pacman实际位置的距离传感器数据

    返回:
    weight: float, 粒子的权重值，使用指数衰减函数计算
    """
	# 使用欧氏距离和指数衰减函数计算权重
	weight = np.exp(-k * np.linalg.norm(gt - estimated, ord=2))
	return weight


def resample_particles(walls, particles: List[Particle]):
	"""
    基于粒子权重进行重采样，保留高权重粒子并生成新样本

    参数:
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息
    particles: List[Particle], 上一次采样得到的粒子，按权重从大到小排列

    返回:
    particles: List[Particle], 重采样后的粒子列表
    """
	resampled_particles: List[Particle] = []
	for _ in range(len(particles)):
		resampled_particles.append(Particle(1.0, 1.0, 1.0, 0.0))

	# 计算地图边界
	xmin = np.min(walls, axis=0)[0]
	ymin = np.min(walls, axis=0)[1]
	xmax = np.max(walls, axis=0)[0]
	ymax = np.max(walls, axis=0)[1]

	N = len(particles)
	total = 0

	# 按权重从大到小排序粒子
	particles.sort(key=lambda p: p.weight, reverse=True)

	# 基于权重比例复制高权重粒子并添加随机扰动
	for i in range(N):
		num = int(particles[i].weight * N * rate)
		count = 0
		error = 0

		# 尝试生成指定数量的粒子副本
		while count < num:
			if error < MAX_ERROR:
				# 在原粒子位置附近添加高斯噪声
				nx = particles[i].position[0] + np.random.normal(0, pos_std)
				ny = particles[i].position[1] + np.random.normal(0, pos_std)
				ntheta = (particles[i].theta + np.random.normal(0, angle_std)) % (2 * np.pi)
				tag = True
				# 检查新位置是否与墙壁碰撞
				for wall in walls:
					wx = wall[0]
					wy = wall[1]
					if circle_square_collision([nx, ny], 0.25, [wx, wy], 1):
						tag = False

				error += 1

				if tag:
					resampled_particles[total] = Particle(nx, ny, ntheta, 1 / N)
					count += 1
					total += 1
					error = 0

				if total >= N:
					break
			else:
				# 如果尝试次数过多仍未找到有效位置，生成随机粒子
				error = 0
				resampled_particles[total] = generate_uniform_particles(walls, 1)[0]
				count += 1
				total += 1

				if total >= N:
					break
		if total >= N:
			break

	# 用随机生成的粒子填充剩余位置
	while total < N:
		x = np.random.uniform(xmin, xmax)
		y = np.random.uniform(ymin, ymax)
		theta = np.random.uniform(0, 2 * np.pi)
		tag = True

		for wall in walls:
			wx = wall[0]
			wy = wall[1]
			if circle_square_collision([x, y], 0.25, [wx, wy], 1):
				tag = False

		if tag:
			resampled_particles[total] = Particle(x, y, theta, 1 / N)
			total += 1

	return resampled_particles


def apply_state_transition(p: Particle, traveled_distance, dtheta):
	"""
    根据Pacman的运动更新粒子状态

    参数:
    p: 采样的粒子
    traveled_distance: float, Pacman移动的距离
    dtheta: float, Pacman运动方向的改变量

    返回:
    particle: 更新位置和方向后的粒子
    """
	x, y = p.position
	theta = p.theta

	# 计算新的方向角（考虑周期性边界）
	ntheta = (theta + dtheta) % (2 * np.pi)

	# 根据距离和方向更新位置
	nx = x + np.cos(ntheta) * traveled_distance
	ny = y + np.sin(ntheta) * traveled_distance

	# 更新粒子状态
	p.position[0] = nx
	p.position[1] = ny
	p.theta = ntheta

	return p


def get_estimate_result(particles: List[Particle]):
	"""
    根据粒子集合估计Pacman的最终位置

    参数:
    particles: List[Particle], 全部采样粒子

    返回:
    final_result: Particle, 权重最高的粒子作为最终估计结果
    """
	# 简单地选择权重最高的粒子作为估计结果
	final_result = particles[0]
	# 用平均的话效果不好
	return final_result