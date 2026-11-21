"""Gymnasium environment for 2D submarine cave navigation (context.md, sections 5 and 6).

Observation (float32), in this order:
    sonar ranges / max_range            n_beams values in [0, 1]
    surge speed u / u_max               [-1, 1]
    yaw rate r / r_max                  [-1, 1]
    goal cue ("bearing" mode only)      cos(bearing), sin(bearing) in the body frame,
                                        straight-line distance / map diagonal in [0, 1]
    previous action                     [thrust, yaw_cmd] in [-1, 1]
With the defaults that is 32 + 2 + 3 + 2 = 39 values (36 with goal_cue="none").
If the sonar pings slower than every step (sonar.ping_interval > dt), the
scan is held between pings and its age / ping_interval is appended.

sensors=("sonar",) (default): the observation is that flat vector.
With "camera" in sensors the observation is a dict:
    "camera": (1, n_pixels) image in [0, 1] (1 channel, for a Conv1d)
    "state":  the vector above, without the sonar part if "sonar" is not in sensors.

Action: [thrust, yaw rate command] in [-1, 1]^2.

Reward per step:
    k_prog * (d_prev - d_now) - k_time - k_smooth * ||a - a_prev||^2 - k_prox * proximity
    - k_energy * (energy used this step, kJ)        [off by default]
    + goal bonus on reaching the goal, - collision penalty on collision
where d is the geodesic distance to the goal. Collision and goal end the
episode (terminated); hitting max_steps truncates it.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from submarine.world import generate_cave, test_seeds, train_seeds
from submarine.sim import Simulator, make_albedo, make_turbidity, wrap_angle

from .collision import resolve_contact
from .failures import FailureProcess
from .navigation import JunctionDetector, PathIntegrator
from .supervisor import SafetySupervisor

from .config import EnvConfig, coerce_config


class CaveEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, config: EnvConfig | dict | str | None = None, render_mode: str | None = None):
        self.cfg = coerce_config(config)
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode
        self._difficulty = self.cfg.difficulty

        c = self.cfg
        self._ping_age = c.sonar.ping_interval > c.vehicle.dt + 1e-9
        n = c.sonar.n_beams if c.use_sonar else 0
        low = [np.zeros(n), -np.ones(2)]
        high = [np.ones(n), np.ones(2)]
        if c.goal_cue == "bearing":
            low.append(np.array([-1.0, -1.0, 0.0]))
            high.append(np.ones(3))
        low.append(-np.ones(2))
        high.append(np.ones(2))
        if self._ping_age and c.use_sonar:
            low.append(np.zeros(1))
            high.append(np.ones(1))
        if c.collision.mode == "damage":
            low.append(np.zeros(2))                   # touch sensor (contact this step), hull integrity / 100
            high.append(np.ones(2))
        self._health_obs = c.failures.enabled or c.supervisor.enabled
        if self._health_obs:
            # sensor health flags (sonar, camera if fitted) and supervisor tier / 4; kept before the navigation
            # features, which must stay last (the junction memory reads them from the end)
            k = 2 + int(c.use_camera)
            low.append(np.zeros(k))
            high.append(np.ones(k))
        if c.navigation.enabled:
            if not c.use_sonar or c.sonar.fov_deg < 360:
                raise ValueError("navigation features need the 360-degree sonar")
            # x, y (scaled, clipped to +-2), cos/sin heading, junction, dead end, write event
            low.append(np.array([-2, -2, -1, -1, 0, 0, 0], dtype=float))
            high.append(np.array([2, 2, 1, 1, 1, 1, 1], dtype=float))
        state_space = spaces.Box(np.concatenate(low).astype(np.float32),
                                 np.concatenate(high).astype(np.float32), dtype=np.float32)
        if c.use_camera:
            cam = spaces.Box(0.0, 1.0, shape=(1, c.camera.n_pixels), dtype=np.float32)
            self.observation_space = spaces.Dict({"camera": cam, "state": state_space})
        else:
            self.observation_space = state_space
        self._state_space = state_space
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        pool = train_seeds if self.cfg.seed_split == "train" else test_seeds
        self._seed_pool = np.array(pool(self.cfg.n_caves))
        self._cache: OrderedDict[tuple[int, float], object] = OrderedDict()
        self.sim: Simulator | None = None
        self._viewer = None
        self._clock = None

    # ------------------------------------------------------------------ API

    def set_difficulty(self, lo: float, hi: float | None = None):
        """Change the difficulty range for future episodes (curriculum; works through VecEnv.env_method)."""
        hi = lo if hi is None else hi
        if not 0.0 <= lo <= hi <= 1.0:
            raise ValueError(f"bad difficulty range ({lo}, {hi})")
        self._difficulty = (float(lo), float(hi))

    def set_layouts(self, layouts):
        """Change the cave layouts sampled in future episodes (curriculum; works through VecEnv.env_method)."""
        self.cfg = replace(self.cfg, layout=tuple(layouts) if not isinstance(layouts, str) else layouts)

    def set_energy_penalty(self, k_energy: float):
        """Change the energy penalty weight for future steps (curriculum)."""
        self.cfg = replace(self.cfg, reward=replace(self.cfg.reward, k_energy=float(k_energy)))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """options: cave_seed, difficulty, current=(cx, cy) override the random choices."""
        super().reset(seed=seed)
        opts = options or {}

        d = opts.get("difficulty")
        if d is None:
            lo, hi = self._difficulty
            d = lo if hi == lo else self.np_random.uniform(lo, hi)
            step = self.cfg.difficulty_step
            if step > 0 and hi > lo:
                d = float(np.clip(round(d / step) * step, lo, hi))
        cave_seed = opts.get("cave_seed")
        if cave_seed is None:
            cave_seed = int(self.np_random.choice(self._seed_pool))
        layout = opts.get("layout")
        if layout is None:
            lays = self.cfg.layout
            layout = lays[0] if len(lays) == 1 else str(self.np_random.choice(lays))
        self.cave, turbidity, albedo = self._get_cave(int(cave_seed), float(d), layout)

        self.sim = Simulator(self.cave, self.cfg.vehicle, self.cfg.sonar,
                             seed=int(self.np_random.integers(2**63 - 1)),
                             camera=self.cfg.camera if self.cfg.use_camera else None,
                             turbidity=turbidity, albedo=albedo)
        self._last = self.sim.reset(current=opts.get("current"))
        s = self._last.state
        self._d_prev = self.sim.geodesic_at(s.x, s.y)
        self._a_prev = np.zeros(2, dtype=np.float32)
        self._t = 0
        self._path_length = 0.0
        self._min_clearance = self._last.clearance
        self._clearance_sum = self._last.clearance
        self._energy = 0.0          # J, metric only (see EnergyConfig)
        self._power = 0.0
        self._trail = [(s.x, s.y)]
        c = self.cfg
        self._failures = FailureProcess(c.failures, np.random.default_rng(int(self.np_random.integers(2**31))),
                                        c.vehicle.dt, c.use_camera, schedule=opts.get("failures"))
        self._supervisor = SafetySupervisor(c.supervisor, c.vehicle)
        if c.supervisor.enabled:
            self._supervisor.update(self._health(), 0.0)
        self._overridden = False
        self._aborted = False
        self._hull, self._contacts, self._contact, self._impact = 100.0, 0, False, 0.0
        self._apply_failures()
        if self.cfg.navigation.enabled:
            self._nav = PathIntegrator(self.cfg.navigation, np.random.default_rng(int(self.np_random.integers(2**31))))
            self._junctions = JunctionDetector(self.cfg.navigation)
            self._nav_origin = (s.x, s.y, s.psi)
            self._update_nav_flags()
        if self.render_mode == "human":
            self.render()
        return self._obs(), self._info(terms={})

    def step(self, action):
        if self.sim is None:
            raise RuntimeError("call reset() before step()")
        a_policy = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)
        prev = self._last.state
        c = self.cfg
        if c.supervisor.enabled:                      # the action actually sent to the thrusters
            self._supervisor.update(self._health(), c.vehicle.dt)
            a = self._supervisor.filter(a_policy, prev.u)
        else:
            a = a_policy
        self._overridden = bool(np.any(np.abs(a - a_policy) > 1e-6))
        self._last = st = self.sim.step(a)
        damage = 0.0
        self._contact, self._impact = False, 0.0
        if c.collision.mode == "damage":
            hit = resolve_contact(self.sim, c.collision, (prev.x, prev.y))
            if hit is not None:                       # bumped: pushed back out, slowed, hull damaged
                self._last = st = self.sim.refresh()
                damage = min(hit["damage"], self._hull)
                self._hull -= damage
                self._contacts += 1
                self._contact, self._impact = True, hit["impact_speed"]
        if c.failures.enabled or self._failures.schedule or self._failures.active:
            self._failures.step()
        self._apply_failures()
        s = st.state
        self._t += 1
        self._path_length += float(np.hypot(s.x - prev.x, s.y - prev.y))
        self._min_clearance = min(self._min_clearance, st.clearance)
        self._clearance_sum += st.clearance
        ec = self.cfg.energy
        self._power = (ec.p_surge * abs(float(a[0])) ** ec.exponent
                       + ec.p_yaw * abs(float(a[1])) ** ec.exponent + ec.p_hotel)
        self._energy += self._power * self.cfg.vehicle.dt
        self._trail.append((s.x, s.y))
        if self.cfg.navigation.enabled:
            self._nav.step(s.u, s.r, self.cfg.vehicle.dt)
            self._update_nav_flags()

        rc = self.cfg.reward
        d_now = self.sim.geodesic_at(s.x, s.y)
        if not np.isfinite(d_now):          # centre is deep inside the infeasible margin (only at collision)
            d_now = self._d_prev
        terms = {
            "progress": rc.k_prog * (self._d_prev - d_now),
            "time": -rc.k_time,
            "smooth": -rc.k_smooth * float(np.sum((a - self._a_prev) ** 2)),
            "proximity": -rc.k_prox * max(0.0, rc.prox_dist - st.clearance) / rc.prox_dist if rc.k_prox else 0.0,
            "terminal": 0.0,
        }
        if rc.k_energy:
            # only added when enabled, so the reward of existing configs stays bit-identical
            terms["energy"] = -rc.k_energy * self._power * self.cfg.vehicle.dt / 1000.0
        if c.supervisor.enabled:
            terms["supervisor"] = -c.supervisor.override_penalty * float(np.abs(a - a_policy).sum())
        if c.collision.mode == "damage":
            terms["damage"] = -c.collision.damage_reward * damage
            collided = self._hull <= 1e-9              # hull breached
        else:
            collided = st.collided
        reached = not collided and self.sim.goal_distance() < self.cfg.goal_radius
        self._aborted = c.supervisor.enabled and not collided and not reached and self._supervisor.tier == 4
        if collided:
            terms["terminal"] = -(c.collision.breach_penalty if c.collision.mode == "damage" else rc.collision)
        elif reached:
            terms["terminal"] = rc.goal
        elif self._aborted:
            terms["terminal"] = -c.supervisor.abort_penalty
        terminated = collided or reached or self._aborted
        truncated = not terminated and self._t >= self.cfg.max_steps

        self._d_prev = d_now
        self._a_prev = a
        reward = float(sum(terms.values()))
        info = self._info(terms, collided=collided, reached=reached, timeout=truncated)
        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None or self.sim is None:
            return None
        from submarine.render.viewer import Viewer   # pygame is only imported when rendering

        if self._viewer is None:
            human = self.render_mode == "human"
            self._viewer = Viewer(self.cave, headless=not human, camera="follow" if human else "full")
        elif self._viewer.cave is not self.cave:
            self._viewer.set_cave(self.cave)
        s = self._last.state
        self._viewer.draw(self.sim, self._last, self._trail, [
            f"seed {self.cave.seed}  d={self.cave.difficulty:.2f}  t={self._t * self.cfg.vehicle.dt:5.1f}s  "
            f"speed {s.u:+.2f} m/s  yaw rate {s.r:+.2f} rad/s",
            f"clearance {self._last.clearance:4.2f} m   geodesic to goal {self._d_prev:5.1f} m",
        ])
        if self.render_mode == "rgb_array":
            return self._viewer.rgb_array()
        import pygame

        pygame.event.pump()
        if self._clock is None:
            self._clock = pygame.time.Clock()
        self._clock.tick(self.metadata["render_fps"])
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ------------------------------------------------------------- helpers

    def _health(self) -> list[bool]:
        """Health flags of the fitted range sensors (what the supervisor and the agent are told)."""
        h = []
        if self.cfg.use_sonar:
            h.append(self._failures.sonar_ok())
        if self.cfg.use_camera:
            h.append(self._failures.camera_ok())
        return h

    def _apply_failures(self):
        """Sensor readings as the agent receives them (degraded by any active failures)."""
        st = self._last
        self._sonar_obs = self._failures.apply_sonar(st.sonar, self.cfg.sonar.max_range)
        self._camera_obs = self._failures.apply_camera(st.camera) if st.camera is not None else None

    def _update_nav_flags(self):
        if not self.cfg.navigation.junction_flags or not self._failures.sonar_ok():   # off / no scan: no events
            self._nav_flags = (0.0, 0.0, 0.0)
            return
        junction, dead_end, event = self._junctions.update(self._sonar_obs, self.sim.sonar.angles,
                                                           (self._nav.x, self._nav.y))
        self._nav_flags = (float(junction), float(dead_end), float(event))

    def nav_true_pose(self):
        """True pose in the navigation frame (start = origin, start heading = 0); for analysis only."""
        x0, y0, p0 = self._nav_origin
        s = self.sim.state
        dx, dy = s.x - x0, s.y - y0
        c, si = np.cos(-p0), np.sin(-p0)
        return dx * c - dy * si, dx * si + dy * c, wrap_angle(s.psi - p0)

    def _get_cave(self, seed: int, d: float, layout: str = "tunnel"):
        """(cave, turbidity field, wall albedo), cached; murk and albedo are fixed per cave."""
        key = (seed, round(d, 6), layout)
        entry = self._cache.get(key)
        if entry is None:
            cave = generate_cave(seed, d, layout=layout)
            entry = (cave, make_turbidity(cave, self.cfg.turbidity), make_albedo(cave))
            self._cache[key] = entry
            if len(self._cache) > self.cfg.cave_cache_size:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        return entry

    def _obs(self):
        st, c = self._last, self.cfg
        s = st.state
        parts = []
        if c.use_sonar:
            parts.append(self._sonar_obs / c.sonar.max_range)
        parts.append([min(1.0, max(-1.0, s.u / c.vehicle.u_max)), min(1.0, max(-1.0, s.r / c.vehicle.r_max))])
        if c.goal_cue == "bearing":
            gx, gy = self.cave.goal
            bearing = wrap_angle(np.arctan2(gy - s.y, gx - s.x) - s.psi)
            w = self.cave.world
            parts.append([np.cos(bearing), np.sin(bearing),
                          min(self.sim.goal_distance() / np.hypot(w.length, w.height), 1.0)])
        parts.append(self._a_prev)
        if self._ping_age and c.use_sonar:
            parts.append([min(st.sonar_age / c.sonar.ping_interval, 1.0)])
        if c.collision.mode == "damage":
            parts.append([float(self._contact), self._hull / 100.0])
        if self._health_obs:
            parts.append([float(h) for h in self._health()] + [self._supervisor.tier / 4.0])
        if c.navigation.enabled:
            nv, k = self._nav, c.navigation.pos_scale
            parts.append([nv.x / k, nv.y / k, np.cos(nv.psi), np.sin(nv.psi), *self._nav_flags])
        sp = self._state_space
        state = np.clip(np.concatenate(parts), sp.low, sp.high).astype(np.float32)
        if not c.use_camera:
            return state
        return {"camera": self._camera_obs.reshape(1, -1).astype(np.float32), "state": state}

    def _info(self, terms: dict, collided=False, reached=False, timeout=False) -> dict:
        s = self._last.state
        return {
            "is_success": reached,
            "collided": collided,
            "timeout": timeout,
            "cave_seed": self.cave.seed,
            "difficulty": self.cave.difficulty,
            "layout": self.cave.config.layout,
            "t": self._t,
            "geodesic": self._d_prev,
            "shortest_path": self.cave.shortest_path_length,
            "path_length": self._path_length,
            "clearance": self._last.clearance,
            "min_clearance": self._min_clearance,
            "mean_clearance": self._clearance_sum / (self._t + 1),
            "energy_j": self._energy,
            "power_w": self._power,
            "position": (s.x, s.y),
            "turbidity": self.sim.turbidity_at(s.x, s.y),
            "aborted": self._aborted,
            "hull": self._hull,
            "contacts": self._contacts,
            "contact": self._contact,
            "impact_speed": self._impact,
            "tier": self._supervisor.tier,
            "overridden": self._overridden,
            "sensor_health": tuple(self._health()),
            "failures": tuple(sorted(self._failures.active)),
            "nav": ({"estimate": (self._nav.x, self._nav.y, self._nav.psi), "flags": self._nav_flags}
                    if self.cfg.navigation.enabled else None),
            "current": tuple(self.sim.current),
            "reward_terms": terms,
        }
