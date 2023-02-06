from pathlib import Path
from queue import PriorityQueue
from typing import Set, Tuple, List

import numpy as np
import numpy.typing as npt

from hw1.utils import neighbors, plot_GVD, PathPlanMode, distance


def cell_to_GVD_gradient_ascent(
	grid: npt.ArrayLike, GVD: Set[Tuple[int, int]], cell: Tuple[int, int]
) -> List[Tuple[int, int]]:
	"""Find the shortest path from any cell in the enviroment to a cell on the
	GVD using gradient ascent.
	Args:
		grid (numpy): NxN numpy array representing the world, with obstacles,
		walls, and the distance from each cell to the obstacles.
		GVD (set[tuple]): A set of tuples containing the cells on the GVD.
		cell (tuple): The starting/ending cell of this path.
	Returns:
		list<tuple>: list of tuples of the path.
	"""

	path = [cell]
	# TODO: Implement this method
	cur_point = cell
	while cur_point not in GVD:
		temp = cur_point
		for i, j in neighbors(grid, cur_point[0], cur_point[1]):
			if grid[i, j] > grid[temp[0], temp[1]]:
				temp = (i, j)
		cur_point = temp
		path.append(cur_point)
	return path


def cell_to_GVD_a_star(
	grid: npt.ArrayLike, GVD: Set[Tuple[int, int]], cell: Tuple[int, int], 
	goal: Tuple[int, int]
) -> List[Tuple[int, int]]:
	"""Find the shortest path from any cell in the enviroment to the GVD using
	A* with L2 distance heurstic.
	Args:
		grid (numpy): NxN numpy array representing the world, with obstacles,
		walls, and the distance from each cell to the obstacles.
		GVD (set<tuple>): A set of tuples containing the cells on the GVD.
		cell (tuple): The starting/ending cell of this path.
	Returns:
		list[tuple], dict, list[int]: list of tuples of the path, and the reached 
		dictionary, and the list of frontier sizes. 
	"""

	# define a priority queue
	frontier = PriorityQueue()
	frontier.put((0, cell))
	frontier_size = []

	# construct a reached table using python dictionary. The key is the (x, y)
	# tuple of the cell position, the value is dictiionary with the cell's cost,
	# and cell parent.
	reached = {cell: {"cost": 0, "parent": None}}

	while not frontier.empty():
		frontier_size.append(frontier.qsize())
		_, current = frontier.get()
		if current in GVD:
			cur_pointer, path = current, [current]
			while reached[cur_pointer]['parent'] != cell:
				path.insert(0, reached[cur_pointer]['parent'])
				cur_pointer = reached[cur_pointer]['parent']
			return path, reached, frontier_size
		for (i, j) in neighbors(grid, *current):
			g = reached[current]['cost'] + distance([current], [(i, j)])
			if (i, j) not in reached or g < reached[(i, j)]['cost']:
				reached[(i, j)] = {"cost":g, "parent": current}
				new_priority = g + distance([(i, j)], [goal])
				frontier.put((new_priority,(i, j)))

	# TODO: implement this to use the reached table (back pointers) to find
	# the path once you have reached a cell on the GVD.
	path = None
	return path, reached, frontier_size


def GVD_path(
	grid: npt.ArrayLike,
	GVD: Set[Tuple[int, int]],
	A: Tuple[int, int],
	B: Tuple[int, int],
	mode: PathPlanMode
) -> List[Tuple[int, int]]:
	"""Find the shortest path between two points on the GVD using
	Breadth-First-Search
	Args:
		grid (numpy): NxN numpy array representing the world, with obstacles,
		walls, and the distance from each cell to the obstacles.
		A (tuple): The starting cell of the path.
		B (tuple): The ending cell of the path.
	Returns:
		list[tuple], dict, list[int]: return the path, pointers, and frontier 
		size array. 
	"""

	# the set of cells on the GVD
	GVD = set(GVD)

	# the set of cells on the current frontier
	frontier = [A]

	# back pointers to find the path once reached the goal B. The keys
	# should both be tuples of cell positions (x, y)
	pointers = {A:A}

	# the length of the frontier array, update this variable at each step. 
	frontier_size = []

	while frontier:
		# TODO:implement this
		frontier_size.append(len(frontier))
		i, j = frontier.pop(0) if mode == PathPlanMode.BFS else frontier.pop()
		for child in neighbors(grid, i, j):
			if child == B:
				cur_pointer, path = (i, j), [B]
				while pointers[cur_pointer] != A:
					path.insert(0, pointers[cur_pointer])
					cur_pointer = pointers[cur_pointer]
				return path, pointers, frontier_size
			if child not in pointers and child in GVD:
				pointers[child] = (i, j)
				frontier.append(child)
	return None, None, None


def compute_path(
	grid,
	GVD: set[tuple],
	start: tuple,
	goal: tuple,
	outmode: PathPlanMode = PathPlanMode.GRAD,
	inmode: PathPlanMode = PathPlanMode.DFS):

	""" Compute the path on the grid from start to goal using the methods
	implemented in this file. 
	Returns:
		list: a list of tuples represent the planned path. 
	"""

	if outmode == PathPlanMode.GRAD:
		start_path = cell_to_GVD_gradient_ascent(grid, GVD, start)
		end_path = list(reversed(cell_to_GVD_gradient_ascent(grid, GVD, goal)))
	else:
		start_path = cell_to_GVD_a_star(grid, GVD, start, goal)[0]
		end_path = list(reversed(cell_to_GVD_a_star(grid, GVD, goal, start)[0]))
	mid_path, reached, frontier_size = GVD_path(
		grid, GVD, start_path[-1], end_path[0], inmode)
	return start_path + mid_path[1:-1] + end_path


def test_world(
	world_id, 
	start, 
	goal,
	outmode: PathPlanMode = PathPlanMode.GRAD,
	inmode: PathPlanMode = PathPlanMode.DFS,
	world_dir="worlds"):

	print(f"Testing world {world_id} with modes {inmode} and {outmode}")
	grid = np.load(f"{world_dir}/world_{world_id}.npy")
	GVD = set([tuple(cell) for cell in np.load(
		f"{world_dir}/world_{world_id}_gvd.npy")])
	path = compute_path(grid, GVD, start, goal, outmode=outmode, inmode=inmode)
	print(f"Path length: {len(path)} steps")
	plot_GVD(grid, world_id, GVD, path)
