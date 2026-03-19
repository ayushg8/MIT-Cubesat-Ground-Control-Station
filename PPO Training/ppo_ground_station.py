import numpy as np
from stable_baselines3 import PPO

VIEW_SIZE = 11
MAX_STEPS = 250

def load_ppo_planner(model_path="lunar_ppo_planner.zip"):
    return PPO.load(model_path)

def ppo_plan_route(model, cost_map, impassable_map, start, goal):
    gs = cost_map.shape[0]
    half = VIEW_SIZE // 2
    MOVES = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
    DISTS = [1.0, 1.414, 1.0, 1.414, 1.0, 1.414, 1.0, 1.414]

    cost_pad = np.pad(cost_map, half, mode="constant", constant_values=1.0)
    imp_pad = np.pad(impassable_map, half, mode="constant", constant_values=1.0)

    pos = np.array(start)
    path = [tuple(pos)]
    total_cost = 0.0
    max_dist = np.sqrt(2) * gs

    for step in range(MAX_STEPS):
        r, c = pos
        rp, cp = r + half, c + half
        cost_view = cost_pad[rp-half:rp+half+1, cp-half:cp+half+1]
        imp_view = imp_pad[rp-half:rp+half+1, cp-half:cp+half+1]

        dist = np.sqrt((goal[0]-r)**2 + (goal[1]-c)**2)
        dist_ch = np.full((VIEW_SIZE, VIEW_SIZE), dist / max_dist, dtype=np.float32)
        dr = (goal[0]-r) / (max_dist+1e-8)
        dc = (goal[1]-c) / (max_dist+1e-8)
        dir_ch = np.full((VIEW_SIZE, VIEW_SIZE), 0.5+0.5*np.arctan2(dr,dc)/np.pi, dtype=np.float32)

        obs = np.stack([cost_view, imp_view, dist_ch, dir_ch], axis=-1).flatten().astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)

        if action < 8:
            mv_dr, mv_dc = MOVES[action]
            nr, nc = r + mv_dr, c + mv_dc
            if 0 <= nr < gs and 0 <= nc < gs and impassable_map[nr, nc] < 0.5:
                pos = np.array([nr, nc])
                path.append(tuple(pos))
                total_cost += cost_map[nr, nc] * DISTS[action]

        if np.sqrt((pos[0]-goal[0])**2 + (pos[1]-goal[1])**2) < 1.5:
            return path, {"path_length": len(path), "total_cost": total_cost, "reached_goal": True}

    return path, {"path_length": len(path), "total_cost": total_cost, "reached_goal": False}

# USAGE:
# model = load_ppo_planner("lunar_ppo_planner.zip")
# path, metrics = ppo_plan_route(model, cost_map, impassable_map, (0,0), (30,30))
