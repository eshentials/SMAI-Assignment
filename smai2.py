import random
import matplotlib.pyplot as plt
import networkx as nx
import heapq
import random
import networkx as nx
from collections import deque

#deprecated

class Venue:
    def __init__(self, vid, start, end, genre):
        self.id = vid
        self.start = start
        self.end = end
        self.genre = genre
        self.children = []

def generate_time_feasible_tree(num_nodes=10, num_genres=3, festival_end=100, seed=42):
    """
    Generates a tree of venues with:
      - Node times increasing
      - Exactly one parent per node
      - Every genre appears at least once
    """
    random.seed(seed)
    genres = [f"Genre_{i}" for i in range(1, num_genres + 1)]

    nodes_meta = []  
    MIN_GAP = 1
    MIN_DUR = 1
    MAX_DUR = 4

    current_time = 0
    # Ensure each genre appears at least once
    for g in genres:
        start = current_time + random.randint(MIN_GAP, MIN_GAP + 2)
        dur = random.randint(MIN_DUR, MAX_DUR)
        end = min(start + dur, festival_end)
        nodes_meta.append((start, end, g))
        current_time = start

    # Add remaining nodes randomly
    while len(nodes_meta) < num_nodes:
        gap = random.randint(0, 3)
        start = min(current_time + gap + 1, festival_end - MIN_DUR)
        dur = random.randint(MIN_DUR, MAX_DUR)
        end = min(start + dur, festival_end)
        g = random.choice(genres)
        nodes_meta.append((start, end, g))
        current_time = start

    nodes_meta = sorted(nodes_meta, key=lambda x: x[0])

    venues = []
    entrance = Venue("entrance", 0, 0, "ENTRANCE")
    venues.append(entrance)

    for i, (s, e, g) in enumerate(nodes_meta, start=1):
        venues.append(Venue(f"v{i}", s, e, g))

    # Assign parents based on time feasibility
    G = nx.DiGraph()
    G.add_node(entrance.id, data=(entrance.start, entrance.end, entrance.genre))
    for v in venues[1:]:
        candidates = [p for p in venues if p.end <= v.start]
        parent = random.choice(candidates) if candidates else entrance
        parent.children.append(v)
        G.add_node(v.id, data=(v.start, v.end, v.genre))
        G.add_edge(parent.id, v.id)

    return venues, genres, G

class TomorrowlandSearch:
    def __init__(self, venues, genres, festival_end):
        self.venues = venues
        self.genres = set(genres)
        self.festival_end = festival_end

    def move_gen(self, state):
        (t, u_id, covered, path) = state
        successors = []
        for v in self.venues:
            if v.start >= t and v.end <= self.festival_end and v.id != u_id:
                new_covered = covered | {v.genre}
                new_state = (v.end, v.id, new_covered, path + [v.id])
                successors.append(new_state)
        return successors


    def goal_test(self, state):
        (t, u, covered, path) = state
        return covered == self.genres and t <= self.festival_end

    def heuristic(self, state, htype=2):
        (t, u, covered, path) = state
        genres_left = len(self.genres - covered)
        if htype == 2:
            return genres_left + (t / self.festival_end)
        elif htype == 3:
            return genres_left * (1 + t / self.festival_end)
        else:
            return genres_left

    def best_first_with_trace(self, htype=2, max_expansions=None):
        start = (0, "entrance", set(), [])
        pq = [(self.heuristic(start, htype), start)]
        visited = set()
        expanded = 0
        trace = []

        while pq:
            hval, state = heapq.heappop(pq)
            expanded += 1
            trace.append((expanded, hval))

            if self.goal_test(state):
                return state[3], expanded, trace

            (t, u, covered, path) = state
            key = (u, tuple(sorted(covered)))
            if key in visited:
                continue
            visited.add(key)

            for succ in self.move_gen(state):
                h = self.heuristic(succ, htype)
                heapq.heappush(pq, (h, succ))

            if max_expansions and expanded >= max_expansions:
                break

        return None, expanded, trace

# BFS
def bfs(problem, max_expansions=None):
    start = (0, "entrance", set(), [])
    q = deque([start])
    visited = set()
    expanded = 0
    trace = []

    while q:
        state = q.popleft()
        expanded += 1
        trace.append((expanded, len(state[2])))

        if problem.goal_test(state):
            return state[3], expanded, trace

        (t, u, covered, path) = state
        key = (u, tuple(sorted(covered)))
        if key in visited:
            continue
        visited.add(key)

        for succ in problem.move_gen(state):
            q.append(succ)

        if max_expansions and expanded >= max_expansions:
            break

    return None, expanded, trace


# DFS
def dfs(problem, max_expansions=None):
    start = (0, "entrance", set(), [])
    stack = [start]
    visited = set()
    expanded = 0
    trace = []

    while stack:
        state = stack.pop()
        expanded += 1
        trace.append((expanded, len(state[2])))

        if problem.goal_test(state):
            return state[3], expanded, trace

        (t, u, covered, path) = state
        key = (u, tuple(sorted(covered)))
        if key in visited:
            continue
        visited.add(key)

        for succ in problem.move_gen(state):
            stack.append(succ)

        if max_expansions and expanded >= max_expansions:
            break

    return None, expanded, trace


# -----------------------------
# Plot search traces
# -----------------------------
def plot_search_traces(trace_bfs, trace_dfs, trace_h2, trace_h3):
    plt.figure(figsize=(9,6))
    if trace_bfs: x, y = zip(*trace_bfs); plt.plot(x, y, label="BFS", color="green")
    if trace_dfs: x, y = zip(*trace_dfs); plt.plot(x, y, label="DFS", color="orange")
    if trace_h2:  x, y = zip(*trace_h2);  plt.plot(x, y, label="Best-First H2", color="red")
    if trace_h3:  x, y = zip(*trace_h3);  plt.plot(x, y, label="Best-First H3", color="blue")
    plt.xlabel("Number of Nodes Expanded")
    plt.ylabel("Progress (genres covered or heuristic value)")
    plt.title("Search Strategies on Tomorrowland Tree")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# -----------------------------
# Visualize solution path on tree
# -----------------------------
def highlight_solution_path(G, path):
    plt.figure(figsize=(12,9))
    pos = nx.spring_layout(G, seed=42, k=0.5)

    node_colors = []
    for n in G.nodes:
        if n in path:
            node_colors.append("red")
        else:
            node_colors.append("lightblue")

    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=700, font_size=8, arrowsize=12)
    plt.title("Tree with Solution Path Highlighted")
    plt.show()




# -----------------------------
# Plot utilities
# -----------------------------
def plot_tree_graph(G):
    plt.figure(figsize=(12, 9))
    pos = nx.spring_layout(G, seed=42, k=0.5)
    labels = {n: f"{G.nodes[n]['data'][2]}\n{G.nodes[n]['data'][0]}-{G.nodes[n]['data'][1]}" for n in G.nodes}
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=700, node_color="lightblue", font_size=8, arrowsize=12)
    plt.title("Time-Feasible Tree (genre \\n start-end)")
    plt.show()


def visualize_tree_and_solution(G, solution_path=None, title_suffix=""):
    """
    Plots the time-feasible tree and highlights the solution path if provided.
    Each call creates a new figure to avoid overwriting previous plots.
    Red nodes: venues included in the solution path.
    Blue nodes: unvisited venues.
    """
    plt.figure(figsize=(12, 9))  # NEW figure
    pos = nx.spring_layout(G, seed=random.randint(0, 10000), k=0.5)  # different seed each time

    node_colors = ["red" if solution_path and n in solution_path else "lightblue" for n in G.nodes]
    labels = {n: f"{G.nodes[n]['data'][2]}\n{G.nodes[n]['data'][0]}-{G.nodes[n]['data'][1]}" for n in G.nodes}

    nx.draw(G, pos, with_labels=True, labels=labels, node_color=node_colors,
            node_size=700, font_size=8, arrowsize=12)
    plt.title(f"Time-Feasible Tree {title_suffix}")
    plt.tight_layout()
    plt.show()  # force display


# -----------------------------
# Runner
# -----------------------------
if __name__ == "__main__":
    NUM_NODES = 40
    NUM_GENRES = 10
    FESTIVAL_END = 100
    SEED = 42

    venues, genres, G = generate_time_feasible_tree(NUM_NODES, NUM_GENRES, FESTIVAL_END, SEED)
    problem = TomorrowlandSearch(venues, genres, FESTIVAL_END)

    # Run searches
    path_bfs, _, _ = bfs(problem)
    path_dfs, _, _ = dfs(problem)
    path_h2, _, _ = problem.best_first_with_trace(htype=2)
    path_h3, _, _ = problem.best_first_with_trace(htype=3)

    # Separate figures for each path
    if path_bfs:
        visualize_tree_and_solution(G, solution_path=path_bfs, title_suffix="(BFS Path)")

    if path_dfs:
        visualize_tree_and_solution(G, solution_path=path_dfs, title_suffix="(DFS Path)")

    if path_h2:
        visualize_tree_and_solution(G, solution_path=path_h2, title_suffix="(Best-First H2 Path)")

    if path_h3:
        visualize_tree_and_solution(G, solution_path=path_h3, title_suffix="(Best-First H3 Path)")
