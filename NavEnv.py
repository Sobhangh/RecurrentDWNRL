from __future__ import annotations

from collections import deque
import queue
import queue
from typing import Optional

import numpy as np
from collections import deque

try:
	import gymnasium as gym
	from gymnasium import spaces
except ImportError:  # pragma: no cover
	import gym
	from gym import spaces


class GridNavEnv(gym.Env):
	"""
	Square grid navigation environment.

	- Start: top-left corner (0, 0)
	- Goal: bottom-right corner (N-1, N-1)
	- Obstacles: 20% of cells sampled randomly (excluding start and goal)
	- Actions: 0=up, 1=right, 2=down, 3=left
	- Observation: adjacent cell states in [top, right, bottom, left]
	  with encoding 0=no cell, 1=free, 2=obstacle, 3=goal
	"""

	metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

	def __init__(self, dimension: int = 10, render_mode: Optional[str] = None):
		super().__init__()

		if dimension < 2:
			raise ValueError("dimension must be >= 2")

		if render_mode is not None and render_mode not in self.metadata["render_modes"]:
			raise ValueError(f"Unsupported render_mode: {render_mode}")

		self.dimension = dimension
		self.render_mode = render_mode

		self.start_pos = (0, 0)
		self.goal_pos = (self.dimension - 1, self.dimension - 1)

		self.action_space = spaces.Discrete(4)
		self.observation_space = spaces.MultiDiscrete([4, 4, 4, 4])

		self.agent_pos = self.start_pos
		self.obstacle_grid = np.zeros((self.dimension, self.dimension), dtype=np.uint8)
		self.obstacles = np.empty((0, 2), dtype=np.int32)
		self.episode_return = 0.0
		self.episode_length = 0
		self.max_episode_length = 100  # Maximum steps per episode

		self._generate_obstacles()

	def check_path_exists(self):
		
		visited = np.zeros((self.dimension, self.dimension), dtype=bool)
		queue = deque([self.start_pos])
		visited[self.start_pos] = True
		while queue:
			r, c = queue.popleft()
			if (r, c) == self.goal_pos:
				return True
			for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
				nr, nc = r + dr, c + dc
				if (
					0 <= nr < self.dimension
					and 0 <= nc < self.dimension
					and not visited[nr, nc]
					and self.obstacle_grid[nr, nc] == 0
				):
					visited[nr, nc] = True
					queue.append((nr, nc))
		return False
            
	def _generate_obstacles(self) -> None:
		self.obstacle_grid.fill(0)

		total_cells = self.dimension * self.dimension
		obstacle_count = int(0.2 * total_cells)

		candidates = [
			(r, c)
			for r in range(self.dimension)
			for c in range(self.dimension)
			if (r, c) != self.start_pos and (r, c) != self.goal_pos
		]

		obstacle_count = min(obstacle_count, len(candidates))
		if obstacle_count == 0:
			self.obstacles = np.empty((0, 2), dtype=np.int32)
			return

		no_path = True
		while no_path:
			self.obstacle_grid.fill(0)
			sampled_indices = self.np_random.choice(len(candidates), size=obstacle_count, replace=False)
			sampled = np.array([candidates[i] for i in sampled_indices], dtype=np.int32)

			self.obstacles = sampled
			self.obstacle_grid[sampled[:, 0], sampled[:, 1]] = 1
			no_path = not self.check_path_exists()

	def _adjacent_observation(self) -> np.ndarray:
		r, c = self.agent_pos
		directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # top, right, bottom, left
		obs = []

		for dr, dc in directions:
			nr, nc = r + dr, c + dc

			if not (0 <= nr < self.dimension and 0 <= nc < self.dimension):
				obs.append(0)
			elif (nr, nc) == self.goal_pos:
				obs.append(3)
			elif self.obstacle_grid[nr, nc] == 1:
				obs.append(2)
			else:
				obs.append(1)

		return np.array(obs, dtype=np.int64)

	def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
		super().reset(seed=seed)
		self.agent_pos = self.start_pos
		self.episode_return = 0.0
		self.episode_length = 0
		self._generate_obstacles()

		observation = self._adjacent_observation()
		info = {"agent_pos": self.agent_pos, "goal_pos": self.goal_pos}

		if self.render_mode == "human":
			self.render()

		return observation, info

	def step(self, action: int):
		if not self.action_space.contains(action):
			raise ValueError(f"Invalid action: {action}")

		r, c = self.agent_pos

		if action == 0:
			nr, nc = r - 1, c
		elif action == 1:
			nr, nc = r, c + 1
		elif action == 2:
			nr, nc = r + 1, c
		else:  # action == 3
			nr, nc = r, c - 1

		reward = -0.01
		terminated = False
		truncated = False

		if not (0 <= nr < self.dimension and 0 <= nc < self.dimension):
			nr, nc = r, c
			reward = -0.02

		hit_obstacle = self.obstacle_grid[nr, nc] == 1
		if (nr, nc) == self.goal_pos:
			reward = 10.0
			terminated = True
		elif hit_obstacle:
			reward = -0.2
			nr, nc = r, c
			if np.random.random() < 0.2:
				terminated = True
		self.agent_pos = (nr, nc)
		self.episode_return += float(reward)
		self.episode_length += 1

		if self.episode_length >= self.max_episode_length:
			truncated = True

		observation = self._adjacent_observation()
		info = {
			"agent_pos": self.agent_pos,
			"hit_obstacle": hit_obstacle,
		}

		# if terminated or truncated:
		# 	info["episode"] = {
		# 		"r": self.episode_return,
		# 		"l": self.episode_length,
		# 	}

		if self.render_mode == "human":
			self.render()

		return observation, reward, terminated, truncated, info

	def _build_rgb_frame(self, cell_size: int = 40) -> np.ndarray:
		h = self.dimension * cell_size
		w = self.dimension * cell_size
		frame = np.ones((h, w, 3), dtype=np.uint8) * 255

		# Paint obstacles red.
		for r, c in self.obstacles:
			y0, y1 = r * cell_size, (r + 1) * cell_size
			x0, x1 = c * cell_size, (c + 1) * cell_size
			frame[y0:y1, x0:x1] = np.array([220, 40, 40], dtype=np.uint8)

		# Paint goal green.
		gr, gc = self.goal_pos
		gy0, gy1 = gr * cell_size, (gr + 1) * cell_size
		gx0, gx1 = gc * cell_size, (gc + 1) * cell_size
		frame[gy0:gy1, gx0:gx1] = np.array([40, 180, 40], dtype=np.uint8)

		# Paint agent blue.
		ar, ac = self.agent_pos
		ay0, ay1 = ar * cell_size, (ar + 1) * cell_size
		ax0, ax1 = ac * cell_size, (ac + 1) * cell_size
		frame[ay0:ay1, ax0:ax1] = np.array([50, 90, 230], dtype=np.uint8)

		# Draw grid lines in gray.
		for i in range(self.dimension + 1):
			y = i * cell_size
			x = i * cell_size
			frame[max(y - 1, 0):y + 1, :] = 150
			frame[:, max(x - 1, 0):x + 1] = 150

		return frame

	def render(self):
		frame = self._build_rgb_frame()

		if self.render_mode == "rgb_array":
			return frame

		if self.render_mode == "human":
			try:
				import matplotlib.pyplot as plt
			except ImportError as exc:  # pragma: no cover
				raise ImportError("matplotlib is required for render_mode='human'") from exc

			if not hasattr(self, "_fig") or self._fig is None:
				self._fig, self._ax = plt.subplots(figsize=(6, 6))
				self._ax.axis("off")
				self._im = self._ax.imshow(frame)
				plt.tight_layout()
				plt.show(block=False)
			else:
				self._im.set_data(frame)

			self._ax.set_title("Grid Navigation")
			self._fig.canvas.draw_idle()
			plt.pause(1 / self.metadata["render_fps"])

		return None

	def close(self):
		if hasattr(self, "_fig") and self._fig is not None:
			try:
				import matplotlib.pyplot as plt

				plt.close(self._fig)
			except Exception:
				pass

			self._fig = None
			self._ax = None
			self._im = None

