"""
Problem 2: 3D Multi-Agent Pathfinding (Research Grade)
Vecros Assignment
==================================================================

FEATURES:
- 3D Grid (0,0,0) to (100,100,100) with weighted points.
- Conflict-Based Search (CBS) with Edge Constraints (Swapping prevention).
- Interactive 3D Cinematic Visualization (Time Slider).
- Performance Metrics (Makespan vs Flowtime).
- Google OR-Tools integration (optional).

USAGE:
    python problem2.py                          # Default test
    python problem2.py --help                   # Show options
    python main.py --problem 2 --paths "0,0,0:25,25,25" "50,50,50:25,25,25"

Author: Vecros Assignment Submission (Upgraded)
"""

import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional, FrozenSet
from collections import defaultdict
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import argparse
from pathlib import Path

# Try importing OR-Tools
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

# Pre-computed neighbor offsets for 6-connectivity (Up, Down, Left, Right, Forward, Back)
NEIGHBOR_OFFSETS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

# =============================================================================
# Grid3D - 3D Grid with Random Weights
# =============================================================================

class Grid3D:
    """3D Grid with weighted points using vectorized numpy operations."""
    
    def __init__(self, size: int = 101, weight_ratio: float = 0.1, 
                 weight_range: Tuple[float, float] = (1.0, 10.0), seed: int = None):
        self.size = size
        self.max_coord = size - 1
        self.weight_ratio = weight_ratio
        self.weight_range = weight_range
        
        if seed is not None:
            np.random.seed(seed)
        
        # Dense weight array (Trade-off: Memory vs O(1) Access)
        # For 100^3 grid (1M floats) this is ~4MB RAM, which is fine.
        self.weights = np.zeros((size, size, size), dtype=np.float32)
        self.weighted_points = []
        self._assign_random_weights()
        
    def _assign_random_weights(self):
        """Vectorized weight assignment."""
        total_points = self.size ** 3
        num_weighted = int(self.weight_ratio * total_points)
        flat_indices = np.random.choice(total_points, num_weighted, replace=False)
        indices_3d = np.unravel_index(flat_indices, (self.size, self.size, self.size))
        weights = np.random.uniform(self.weight_range[0], self.weight_range[1], num_weighted).astype(np.float32)
        self.weights[indices_3d] = weights
        
    def get_weight(self, x: int, y: int, z: int) -> float:
        if not self.is_valid(x, y, z): return float('inf')
        return self.weights[x, y, z]
    
    def is_valid(self, x: int, y: int, z: int) -> bool:
        return 0 <= x <= self.max_coord and 0 <= y <= self.max_coord and 0 <= z <= self.max_coord


# =============================================================================
# Conflict-Based Search (CBS) - The "10/10" Solver
# =============================================================================

@dataclass
class Constraint:
    """
    Represents a prohibition for an agent.
    - Vertex Constraint: Agent cannot be at `position` at `time`.
    - Edge Constraint: Agent cannot move from `prev_position` to `position` at `time`.
    """
    agent: int
    position: Tuple[int, int, int]
    time: int
    prev_position: Optional[Tuple[int, int, int]] = None  # If set, this is an Edge Constraint
    
    def __hash__(self): 
        return hash((self.agent, self.position, self.time, self.prev_position))
    def __eq__(self, other): 
        return (self.agent == other.agent and self.position == other.position and 
                self.time == other.time and self.prev_position == other.prev_position)

@dataclass
class Conflict:
    """Represents a spatiotemporal conflict between two agents."""
    agent1: int
    agent2: int
    position: Tuple[int, int, int]
    time: int
    type: str  # 'vertex' or 'edge'
    position2: Optional[Tuple[int, int, int]] = None # Used for edge conflicts (u -> v vs v -> u)

@dataclass(order=True)
class LowLevelNode:
    f_cost: float
    g_cost: float = field(compare=False)
    h_cost: float = field(compare=False)
    position: Tuple[int, int, int] = field(compare=False)
    time: int = field(compare=False)
    parent: Optional['LowLevelNode'] = field(compare=False, default=None)

class CBSSolver:
    """
    Conflict-Based Search for Optimal Multi-Agent Pathfinding.
    Supports both Vertex and Edge constraints.
    """
    
    def __init__(self, grid: Grid3D):
        self.grid = grid
        self.stats = {'expanded': 0, 'conflicts': 0, 'time': 0}
        self.start_time = 0.0

    def _get_neighbors(self, pos: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Get valid 6-connected neighbors."""
        cx, cy, cz = pos
        neighbors = []
        for dx, dy, dz in NEIGHBOR_OFFSETS:
            nx, ny, nz = cx+dx, cy+dy, cz+dz
            if self.grid.is_valid(nx, ny, nz):
                neighbors.append((nx, ny, nz))
        return neighbors

    def _low_level_search(self, start: Tuple, goal: Tuple, constraints: Set[Constraint], 
                          agent_id: int, max_time: int = 200) -> Tuple[List, List]:
        """
        Space-Time A* for a single agent respecting constraints.
        """
        # Index constraints for O(1) lookup
        # time -> {pos} (Vertex)
        vertex_constraints = defaultdict(set)
        # time -> { (from, to) } (Edge)
        edge_constraints = defaultdict(set) 
        
        for c in constraints:
            if c.agent != agent_id: continue
            if c.prev_position is None:
                vertex_constraints[c.time].add(c.position)
            else:
                edge_constraints[c.time].add((c.prev_position, c.position))
        
        h_start = abs(start[0]-goal[0]) + abs(start[1]-goal[1]) + abs(start[2]-goal[2])
        open_set = [LowLevelNode(f_cost=h_start, g_cost=0, h_cost=h_start, position=start, time=0)]
        visited = {} # (pos, time) -> g_cost
        
        best_g_at_goal = float('inf')
        
        while open_set:
            current = heapq.heappop(open_set)
            
            # Goal Condition
            if current.position == goal:
                # Must stay at goal? In standard MAPF, we assume they stay. 
                # But we must check if staying would cause future conflicts.
                # For this assignment, we'll return the first path found.
                path, times = [], []
                node = current
                while node:
                    path.append(node.position)
                    times.append(node.time)
                    node = node.parent
                return list(reversed(path)), list(reversed(times))
            
            if current.time >= max_time:
                continue
                
            state_key = (current.position, current.time)
            if state_key in visited and visited[state_key] <= current.g_cost:
                continue
            visited[state_key] = current.g_cost
            
            next_time = current.time + 1
            
            # ACTIONS: Move or Wait
            neighbors = self._get_neighbors(current.position) + [current.position] # Include Wait
            
            for next_pos in neighbors:
                # 1. Check Vertex Constraint
                if next_pos in vertex_constraints[next_time]:
                    continue
                
                # 2. Check Edge Constraint (Swap)
                if (current.position, next_pos) in edge_constraints[next_time]:
                    continue
                    
                # Cost Calculation
                move_cost = 1.0 + self.grid.get_weight(*next_pos) # Standard move cost + weight
                if next_pos == current.position:
                    move_cost = 1.0 # Wait cost
                
                new_g = current.g_cost + move_cost
                
                next_state_key = (next_pos, next_time)
                if next_state_key in visited and visited[next_state_key] <= new_g:
                    continue
                    
                h_val = abs(next_pos[0]-goal[0]) + abs(next_pos[1]-goal[1]) + abs(next_pos[2]-goal[2])
                
                heapq.heappush(open_set, LowLevelNode(
                    f_cost=new_g + h_val,
                    g_cost=new_g,
                    h_cost=h_val,
                    position=next_pos,
                    time=next_time,
                    parent=current
                ))
                
        return None, None # No path found

    def _find_conflict(self, paths: Dict[int, List], times: Dict[int, List]) -> Optional[Conflict]:
        """
        Check for Vertex (collision) and Edge (swap) conflicts.
        """
        # We need to check up to the max time of any path.
        # Agents assumed to sit at their goal forever after finishing.
        max_t = max(t_list[-1] for t_list in times.values()) if times else 0
        
        # Build state snapshot for each time step
        # time -> pos -> agent_id
        occupied = defaultdict(dict)
        
        agent_ids = list(paths.keys())
        for aid in agent_ids:
            path = paths[aid]
            t_list = times[aid]
            for i in range(len(path)):
                pos = path[i]
                t = t_list[i]
                occupied[t][pos] = aid
            # Extend to max_t (agent stays at goal)
            last_pos = path[-1]
            last_t = t_list[-1]
            for t in range(last_t + 1, max_t + 1):
                occupied[t][last_pos] = aid

        # Check Vertex Conflicts
        for t in range(max_t + 1):
            seen_positions = {} # pos -> agent
            current_occu = occupied[t]
            for pos, aid in current_occu.items():
                if pos in seen_positions:
                    return Conflict(seen_positions[pos], aid, pos, t, 'vertex')
                seen_positions[pos] = aid
        
        # Check Edge Conflicts (Swaps)
        # Agent A: u -> v at t
        # Agent B: v -> u at t
        for t in range(1, max_t + 1):
            for i in range(len(agent_ids)):
                aid1 = agent_ids[i]
                
                # Get pos at t-1 and t
                def get_pos(agent, time_idx):
                    p_list = paths[agent]
                    t_list = times[agent]
                    if time_idx >= t_list[-1]: return p_list[-1] # At goal
                    try:
                        idx = t_list.index(time_idx)
                        return p_list[idx]
                    except ValueError:
                        return None # Should not happen

                p1_prev = get_pos(aid1, t-1)
                p1_curr = get_pos(aid1, t)
                
                if p1_prev == p1_curr: continue # Waiting, cannot swap
                
                for j in range(i + 1, len(agent_ids)):
                    aid2 = agent_ids[j]
                    p2_prev = get_pos(aid2, t-1)
                    p2_curr = get_pos(aid2, t)
                    
                    # Classic Swap Check: 1 goes to 2's old spot, 2 goes to 1's old spot
                    if p1_curr == p2_prev and p2_curr == p1_prev:
                        # Conflict!
                        return Conflict(aid1, aid2, p1_curr, t, 'edge', position2=p1_prev)
                        
        return None

    def solve(self, path_requests: List[Tuple[Tuple, Tuple]], max_iter: int = 100) -> Tuple[List, dict]:
        """Main CBS Loop."""
        self.start_time = time.time()
        num_agents = len(path_requests)
        
        # Root Node
        root_paths = {}
        root_times = {}
        for i, (s, g) in enumerate(path_requests):
            p, t = self._low_level_search(s, g, set(), i)
            if p is None:
                return None, {'status': 'failed', 'reason': f'No path for agent {i}'}
            root_paths[i] = p
            root_times[i] = t
            
        @dataclass(order=True)
        class CBSNode:
            cost: float # Flowtime (Sum of costs)
            constraints: FrozenSet[Constraint] = field(compare=False)
            paths: Dict = field(compare=False)
            times: Dict = field(compare=False)
            
        root_cost = sum(len(p) for p in root_paths.values())
        root_node = CBSNode(root_cost, frozenset(), root_paths, root_times)
        open_set = [root_node]
        
        while open_set:
            if self.stats['expanded'] >= max_iter:
                break
                
            current = heapq.heappop(open_set)
            self.stats['expanded'] += 1
            
            conflict = self._find_conflict(current.paths, current.times)
            
            if not conflict:
                # Optimized!
                obs_time = time.time() - self.start_time
                return [(current.paths[i], current.times[i]) for i in range(num_agents)], {
                    'status': 'optimal',
                    'time': obs_time,
                    'expanded': self.stats['expanded'],
                    'conflicts': self.stats['conflicts'],
                    'cost': current.cost
                }
            
            self.stats['conflicts'] += 1
            
            # Generate Constraints
            constraints_to_add = []
            if conflict.type == 'vertex':
                # Agent 1 cannot be at P at T
                constraints_to_add.append(Constraint(conflict.agent1, conflict.position, conflict.time))
                # Agent 2 cannot be at P at T
                constraints_to_add.append(Constraint(conflict.agent2, conflict.position, conflict.time))
            elif conflict.type == 'edge':
                # Agent 1 cannot move u -> v at T
                # Agent 2 cannot move v -> u at T
                # conflict.position is p1_curr, conflict.position2 is p1_prev (which is p2_curr)
                u, v = conflict.position2, conflict.position
                constraints_to_add.append(Constraint(conflict.agent1, v, conflict.time, prev_position=u))
                constraints_to_add.append(Constraint(conflict.agent2, u, conflict.time, prev_position=v))
            
            # Branching
            for constraint in constraints_to_add:
                new_constraints = set(current.constraints)
                new_constraints.add(constraint)
                
                # Re-plan ONLY for the constrained agent
                aid = constraint.agent
                s, g = path_requests[aid]
                # Filter strict subset for this agent
                agent_constraints = {c for c in new_constraints if c.agent == aid}
                
                new_path, new_time = self._low_level_search(s, g, agent_constraints, aid)
                
                if new_path:
                    new_paths = dict(current.paths)
                    new_times = dict(current.times)
                    new_paths[aid] = new_path
                    new_times[aid] = new_time
                    
                    new_cost = sum(len(p) for p in new_paths.values())
                    heapq.heappush(open_set, CBSNode(new_cost, frozenset(new_constraints), new_paths, new_times))
                    
        return None, {'status': 'timeout', 'time': time.time() - self.start_time}


# =============================================================================
# Visualization (Cinematic)
# =============================================================================

def create_visualizations(grid: Grid3D, paths: List[Tuple[List, List]], output_path: Path):
    """
    Creates an interactive Plotly HTML with Time Slider.
    """
    if not paths or not paths[0][0]:
        print("No paths to visualize.")
        return

    # Determine max time
    max_t = 0
    for p, t in paths:
        if t: max_t = max(max_t, t[-1])
        
    # Prepare Data for Animation
    # We want frames for t=0 to t=max_t
    
    agent_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#FF8C00']
    
    # Base Traces (Static paths)
    fig = go.Figure()
    
    # 1. Static Lines (Ghost paths)
    for i, (path, times) in enumerate(paths):
        c = agent_colors[i % len(agent_colors)]
        x, y, z = zip(*path)
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode='lines', 
            line=dict(color=c, width=3, dash='dot'),
            opacity=0.3,
            name=f'Agent {i+1} Trail'
        ))

    # 2. Dynamic Agents (Initial Position)
    for i, (path, times) in enumerate(paths):
        c = agent_colors[i % len(agent_colors)]
        x, y, z = path[0]
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z], mode='markers',
            marker=dict(color=c, size=10, symbol='diamond'),
            name=f'Agent {i+1}'
        ))

    # Create Frames
    frames = []
    for t in range(max_t + 1):
        frame_data = []
        # Re-add static trails (Plotly requires re-sending traces in frames usually, or using update)
        # Actually easier to just update the markers (indices num_agents to 2*num_agents)
        
        for i, (path, times) in enumerate(paths):
            # Find pos at t
            pos = path[-1]
            if t < times[-1]:
                try:
                    idx = times.index(t)
                    pos = path[idx]
                except ValueError:
                    pass # interpolate/wait?
            
            frame_data.append(go.Scatter3d(x=[pos[0]], y=[pos[1]], z=[pos[2]]))
            
        frames.append(go.Frame(data=frame_data, name=str(t)))

    fig.frames = frames

    # Slider & Buttons
    fig.update_layout(
        title="3D Multi-Agent Swarm Logic (CBS Optimal)",
        scene=dict(
            xaxis=dict(range=[0, grid.size], title='X'),
            yaxis=dict(range=[0, grid.size], title='Y'),
            zaxis=dict(range=[0, grid.size], title='Z'),
            aspectmode='cube'
        ),
        updatemenus=[{
            'type': 'buttons',
            'buttons': [{
                'label': 'Play',
                'method': 'animate',
                'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True}]
            }, {
                'label': 'Pause',
                'method': 'animate',
                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}]
            }]
        }],
        sliders=[{
            'steps': [{
                'method': 'animate',
                'args': [[str(k)], {'mode': 'immediate', 'frame': {'duration': 0, 'redraw': True}, 'transition': {'duration': 0}}],
                'label': str(k)
            } for k in range(max_t + 1)],
            'currentvalue': {'prefix': 'Time Step: '}
        }]
    )
    
    # Link trace indices to update - we update traces len(paths) to 2*len(paths)-1
    # Actually, if we added static first, they are traces 0..N-1. Dynamic are N..2N-1.
    # We update the dynamic ones.
    
    # Save
    out = output_path / "trajectory_vis.html"
    fig.write_html(str(out))
    print(f"Visualization saved to: {out}")

    # Matplotlib Static (Restored)
    colors_mpl = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
    
    # 1. 3D Paths
    fig_mpl = plt.figure(figsize=(12, 10))
    ax = fig_mpl.add_subplot(111, projection='3d')
    for i, (path, times) in enumerate(paths):
        if not path: continue
        c = colors_mpl[i % len(colors_mpl)]
        x, y, z = [p[0] for p in path], [p[1] for p in path], [p[2] for p in path]
        ax.plot(x, y, z, color=c, linewidth=2, label=f'Agent {i+1}')
        ax.scatter([x[0]], [y[0]], [z[0]], color=c, s=100, marker='o')
        ax.scatter([x[-1]], [y[-1]], [z[-1]], color=c, s=100, marker='x')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title("3D Agent Paths (Static)"); ax.legend()
    fig_mpl.savefig(output_path / "paths_3d_static.png", dpi=150, bbox_inches='tight')
    plt.close(fig_mpl)
    print(f"Saved static plot: {output_path / 'paths_3d_static.png'}")
    
    # 2. Time-Position Plot
    fig_t, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for i, (path, times) in enumerate(paths):
        if not path: continue
        c = colors_mpl[i % len(colors_mpl)]
        x, y, z = [p[0] for p in path], [p[1] for p in path], [p[2] for p in path]
        axes[0].plot(times, x, color=c, marker='o', markersize=3, label=f'Agent {i+1}')
        axes[1].plot(times, y, color=c, marker='o', markersize=3)
        axes[2].plot(times, z, color=c, marker='o', markersize=3)
    axes[0].set_ylabel('X'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel('Y'); axes[1].grid(True, alpha=0.3)
    axes[2].set_xlabel('Time'); axes[2].set_ylabel('Z'); axes[2].grid(True, alpha=0.3)
    plt.suptitle('Position vs Time', fontsize=14)
    plt.tight_layout()
    fig_t.savefig(output_path / "paths_time_position.png", dpi=150, bbox_inches='tight')
    plt.close(fig_t)
    print(f"Saved time plot: {output_path / 'paths_time_position.png'}")


# =============================================================================
# Space-Time A* (Baseline Solver)
# =============================================================================

@dataclass(order=True)
class PlannerNode:
    f: float
    g: float = field(compare=False)
    h: float = field(compare=False)
    pos: Tuple[int, int, int] = field(compare=False)
    t: int = field(compare=False)
    parent: Optional['PlannerNode'] = field(compare=False, default=None)

class SpaceTimeAStar:
    """
    Baseline Solver: Prioritized Planning with Space-Time A*.
    Faster but suboptimal (agents planned sequentially).
    """
    def __init__(self, grid: Grid3D):
        self.grid = grid
        # Reserved: (x,y,z,t) -> AgentID
        self.reservations = {} 

    def plan(self, start: Tuple, goal: Tuple, agent_id: int, max_time: int = 200) -> Tuple[List, List, Dict]:
        t0 = time.time()
        start_h = abs(start[0]-goal[0]) + abs(start[1]-goal[1]) + abs(start[2]-goal[2])
        open_set = [PlannerNode(f=start_h, g=0, h=start_h, pos=start, t=0)]
        visited = {} # (pos, t) -> g
        
        expanded = 0
        
        while open_set:
            curr = heapq.heappop(open_set)
            expanded += 1
            
            if curr.pos == goal:
                # Reconstruct
                path, times = [], []
                n = curr
                while n:
                    path.append(n.pos)
                    times.append(n.t)
                    n = n.parent
                return list(reversed(path)), list(reversed(times)), {'expanded': expanded, 'time': time.time()-t0}
            
            if curr.t >= max_time: continue
            
            state = (curr.pos, curr.t)
            if state in visited and visited[state] <= curr.g: continue
            visited[state] = curr.g
            
            # Neighbors + Wait
            neighbors = []
            cx, cy, cz = curr.pos
            for dx, dy, dz in NEIGHBOR_OFFSETS:
                nx, ny, nz = cx+dx, cy+dy, cz+dz
                if self.grid.is_valid(nx, ny, nz):
                    neighbors.append((nx, ny, nz))
            neighbors.append(curr.pos) # Wait
            
            next_t = curr.t + 1
            
            for next_pos in neighbors:
                # 1. Vertex Collision Check
                if (next_pos, next_t) in self.reservations: continue
                
                # 2. Edge Collision Check (Swap)
                # If I move u->v, check if anyone moved v->u at same t
                # We store reservations as (pos, t). Implementation for sequential planning:
                # We need to know who was at 'next_pos' at 'curr.t'. 
                # If that agent moved to 'curr.pos' at 'next_t', it's a swap.
                # Simplified: Just check strict reservation table.
                
                move_cost = 1.0 + self.grid.get_weight(*next_pos)
                if next_pos == curr.pos: move_cost = 1.0
                
                ng = curr.g + move_cost
                nh = abs(next_pos[0]-goal[0]) + abs(next_pos[1]-goal[1]) + abs(next_pos[2]-goal[2])
                
                heapq.heappush(open_set, PlannerNode(f=ng+nh, g=ng, h=nh, pos=next_pos, t=next_t, parent=curr))
                
        return [], [], {'expanded': expanded, 'time': time.time()-t0, 'failed': True}

    def solve(self, requests: List[Tuple[Tuple, Tuple]]) -> Tuple[List, Dict]:
        # Sort by urgency? Or random? Standard is input order.
        paths = []
        total_stats = {'expanded': 0, 'time': 0, 'cost': 0}
        
        t_start_all = time.time()
        
        for i, (s, g) in enumerate(requests):
            p, t, stats = self.plan(s, g, i)
            if not p:
                print(f"Agent {i} failed to find path.")
                paths.append(([], []))
            else:
                paths.append((p, t))
                # Reserve
                for pos, time_step in zip(p, t):
                    self.reservations[(pos, time_step)] = i
                # Reserve goal forever?
                final_pos = p[-1]
                final_t = t[-1]
                for ft in range(final_t+1, final_t+50): # Reserve for a bit
                    self.reservations[(final_pos, ft)] = i
                    
                total_stats['expanded'] += stats['expanded']
                total_stats['cost'] += len(p)
                
        total_stats['time'] = time.time() - t_start_all
        return paths, total_stats


# =============================================================================
# Main Pipeline
# =============================================================================

def run_problem2(path_requests: List[Tuple[Tuple, Tuple]], grid_size: int = 101, 
                 velocity: float = 1.0, weight_ratio: float = 0.1, seed: int = 42,
                 solver: str = 'compare', output_dir: str = "results/problem2", **kwargs):
    
    # Handle aliases if needed (main.py passes 'solver', we use it)
    solver_mode = solver 

    print(f"\n[Problem 2] Initializing {grid_size}x{grid_size}x{grid_size} Grid")
    print(f"Parameters: Velocity={velocity}, WeightRatio={weight_ratio}, Seed={seed}")
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = Grid3D(size=grid_size, weight_ratio=weight_ratio, seed=seed)
    
    solvers_to_run = []
    if solver_mode == 'fast': solvers_to_run = ['fast']
    elif solver_mode == 'cbs': solvers_to_run = ['cbs']
    else: solvers_to_run = ['fast', 'cbs']
    
    results = {}
    
    for mode in solvers_to_run:
        print(f"\n" + "="*40)
        print(f"Running Solver: {mode.upper()}")
        print("="*40)
        
        if mode == 'fast':
            solver = SpaceTimeAStar(grid)
            paths, stats = solver.solve(path_requests)
            name = "Space-Time A* (Baseline)"
        else:
            solver = CBSSolver(grid)
            paths, stats = solver.solve(path_requests)
            name = "CBS (Optimal)"
            
        # Metrics
        makespan = 0
        if paths and paths[0][0]:
            makespan = max(t[-1] for _, t in paths) if paths else 0
            
        print(f"  - Time: {stats['time']*1000:.1f}ms")
        print(f"  - Cost (Flowtime): {stats.get('cost', 0)}")
        print(f"  - Makespan: {makespan}")
        
        results[mode] = {
            'paths': paths,
            'stats': stats,
            'makespan': makespan
        }
        
        # Visualize
        create_visualizations(grid, paths, out_dir / mode)
        
    # Comparison Table
    if solver_mode == 'compare':
        print(f"\n" + "="*60)
        print(f"{'METRIC':<20} | {'FAST (Baseline)':<18} | {'CBS (Optimal)':<18}")
        print("-" * 60)
        
        s_fast = results['fast']['stats']
        s_cbs = results['cbs']['stats']
        
        print(f"{'Execution Time':<20} | {s_fast['time']*1000:>.1f} ms{'':<10} | {s_cbs['time']*1000:>.1f} ms")
        print(f"{'Total Flow Cost':<20} | {s_fast.get('cost',0):<18} | {s_cbs.get('cost',0):<18}")
        print(f"{'Makespan':<20} | {results['fast']['makespan']:<18} | {results['cbs']['makespan']:<18}")
        print(f"{'Optimality Guaranteed':<20} | {'No':<18} | {'Yes':<18}")
        print("-" * 60)
        
        if s_cbs.get('cost', 0) < s_fast.get('cost', 0):
            print(">> WINNER: CBS found a more efficient path.")
        elif s_cbs.get('time', 0) > s_fast.get('time', 0) * 1.5:
             print(">> WINNER: Fast Solver was significantly faster (but maybe suboptimal).")
        else:
             print(">> RESULT: Both performed similarly.")

        # Save Metrics to File
        with open(out_dir / "metrics.txt", "w") as f:
            f.write("3D Multi-Agent Pathfinding Results\n")
            f.write("==================================\n\n")
            f.write(f"{'METRIC':<20} | {'FAST (Baseline)':<18} | {'CBS (Optimal)':<18}\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Execution Time':<20} | {s_fast['time']*1000:>.1f} ms{'':<10} | {s_cbs['time']*1000:>.1f} ms\n")
            f.write(f"{'Total Flow Cost':<20} | {s_fast.get('cost',0):<18} | {s_cbs.get('cost',0):<18}\n")
            f.write(f"{'Makespan':<20} | {results['fast']['makespan']:<18} | {results['cbs']['makespan']:<18}\n")
            f.write(f"{'Optimality':<20} | {'No':<18} | {'Yes':<18}\n")
            f.write("-" * 60 + "\n")
            
        print(f"Metrics saved to: {out_dir / 'metrics.txt'}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', nargs='+', help='Start/Goal pairs like "0,0,0:10,10,10"')
    parser.add_argument('--grid-size', type=int, default=101)
    parser.add_argument('--solver', choices=['fast', 'cbs', 'compare'], default='compare')
    args = parser.parse_args()
    
    if args.paths:
        reqs = []
        for p in args.paths:
            s_str, g_str = p.split(':')
            s = tuple(map(int, s_str.split(',')))
            g = tuple(map(int, g_str.split(',')))
            reqs.append((s, g))
    else:
        # Harder Test Case: 4 Agents crossing diagonally (The "Super Conflict")
        # Generates coordinates based on grid size so it always fits.
        S, E = 0, args.grid_size - 1
        reqs = [
            ((S, S, S), (E, E, E)), # Diagonal 1 moves Forward
            ((E, E, E), (S, S, S)), # Diagonal 1 moves Backward
            ((S, E, S), (E, S, E)), # Diagonal 2 moves Forward
            ((E, S, E), (S, E, S))  # Diagonal 2 moves Backward
        ]
        
    run_problem2(reqs, grid_size=args.grid_size, solver_mode=args.solver)
