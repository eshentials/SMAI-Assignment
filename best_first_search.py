from collections import deque
import heapq
import matplotlib.pyplot as plt
import networkx as nx

# -----------------------
# Problem Representation (same as yours)
# -----------------------
class Venue:
    def __init__(self, vid, start, end, genre):
        self.id = vid
        self.start = start
        self.end = end
        self.genre = genre

    def __repr__(self):
        return f"Venue({self.id}, {self.start}-{self.end}, {self.genre})"

class TomorrowlandSearch:
    def __init__(self, venues, genres, festival_end):
        self.venues = venues
        self.genres = set(genres)
        self.festival_end = festival_end

    def move_gen(self, state):
        (t, u, covered, path) = state
        successors = []
        for v in self.venues:
            if v.start >= t and v.end <= self.festival_end:
                new_covered = set(covered) | {v.genre}
                new_state = (v.end, v.id, frozenset(new_covered), path + [v.id])
                successors.append(new_state)
        return successors

    def goal_test(self, state):
        (t, u, covered, path) = state
        return set(covered) == self.genres and t <= self.festival_end

    def heuristic(self, state, htype=1):
        (t, u, covered, path) = state
        genres_left = len(self.genres - set(covered))
        if htype == 2:
            return genres_left + (t / self.festival_end)  # H2
        elif htype == 3:
            return genres_left * (1 + t / self.festival_end)  # H3
        else:
            return genres_left

# -----------------------
# Fixed Best-first search that builds tree with ONLY explored nodes
# -----------------------
def best_first_tree(problem, htype=2):
    start = (0, "entrance", frozenset(), [])  # Use frozenset instead of set
    pq = [(problem.heuristic(start, htype), 0, start, None)]  # Added node_id and parent_id
    visited = set()
    expanded = 0
    trace = []
    tree_edges = []
    node_states = {}
    parent = {}
    node_counter = 0
    goal_node = None
    
    # Add start node
    node_states[node_counter] = start
    node_counter += 1

    while pq:
        hval, current_node_id, state, parent_node_id = heapq.heappop(pq)
        (t, u, covered, path) = state

        # only expand unique states
        key = (t, tuple(sorted(covered)))
        if key in visited:
            continue
        visited.add(key)

        expanded += 1
        trace.append((expanded, hval))
        
        # Record parent relationship only when we actually expand this node
        if parent_node_id is not None:
            parent[current_node_id] = parent_node_id
            tree_edges.append((parent_node_id, current_node_id))

        # goal test
        if problem.goal_test(state):
            goal_node = current_node_id
            return path, expanded, trace, tree_edges, node_states, parent, goal_node

        # expand successors - only add edges for nodes we actually explore
        for succ in problem.move_gen(state):
            h = problem.heuristic(succ, htype)
            
            # Create new node for successor
            succ_node_id = node_counter
            node_states[succ_node_id] = succ
            node_counter += 1
            
            # Add to priority queue with parent relationship
            heapq.heappush(pq, (h, succ_node_id, succ, current_node_id))

    return None, expanded, trace, tree_edges, node_states, parent, goal_node

# -----------------------
# Hierarchical layout (no pygraphviz)
# -----------------------
def hierarchy_pos(G, root, width=1., vert_gap=0.18, vert_loc=0, xcenter=0.5, pos=None, parent=None):
    """
    Recursively position nodes in a hierarchy. G is a DiGraph, root is node id.
    """
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    children = list(G.successors(root))
    if parent is not None and parent in children:
        children.remove(parent)
    if len(children) == 0:
        return pos
    dx = width / len(children) if len(children) > 0 else width
    nextx = xcenter - width/2 + dx/2
    for child in children:
        pos = hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, 
                            vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos, parent=root)
        nextx += dx
    return pos

# -----------------------
# Plotting a traversal tree and highlight solution path
# -----------------------
def plot_traversal_tree(edges, node_states, parent, goal_node, title):
    if not edges:  # Handle case with no edges (single node)
        plt.figure(figsize=(8,6))
        plt.text(0.5, 0.5, "Only root node explored", ha='center', va='center', fontsize=14)
        plt.title(title)
        plt.axis('off')
        plt.show()
        return
        
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # build labels: show venue id (or 'entrance') and covered-count for clarity
    labels = {}
    for nid, state in node_states.items():
        if nid in G.nodes() or nid == 0:  # Only label nodes that are in the graph or root
            t, u, covered_fset, path = state
            label = f"{u}"
            # optionally include covered count for debugging
            label += f"\n({len(covered_fset)})"
            labels[nid] = label

    plt.figure(figsize=(14,8))
    
    # Handle single node case
    if len(G.nodes()) == 0:
        pos = {0: (0.5, 0.5)}
        G.add_node(0)
    else:
        pos = hierarchy_pos(G, 0, width=1.0, vert_gap=0.18, vert_loc=0.9, xcenter=0.5)

    # highlight path nodes & edges
    path_nodes = []
    path_edges = set()
    if goal_node is not None:
        # reconstruct path node ids from goal_node to root
        cur = goal_node
        while True:
            path_nodes.append(cur)
            if cur == 0:
                break
            cur = parent.get(cur, 0)
        path_nodes.reverse()
        for a, b in zip(path_nodes[:-1], path_nodes[1:]):
            path_edges.add((a, b))

    # draw nodes: color path nodes differently
    node_colors = []
    for n in G.nodes():
        if n in path_nodes and n != 0:
            node_colors.append("orange")
        elif n == 0:
            node_colors.append("lightgreen")  # entrance
        else:
            node_colors.append("lightblue")

    # draw edges with color mapping (highlight path edges in red)
    edge_colors = []
    for e in G.edges():
        if e in path_edges:
            edge_colors.append("red")
        else:
            edge_colors.append("gray")

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1400)
    if len(G.edges()) > 0:
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, arrowsize=16, width=1.5)
    nx.draw_networkx_labels(G, pos, labels, font_size=9)

    plt.title(title)
    plt.axis("off")
    plt.show()

# -----------------------
# Test Cases - 4 genres, concurrent events (your updated input)
# -----------------------
genres = [f"genre{i}" for i in range(1, 5)]
festival_end = 12

venues1 = [
    Venue("v1", 1, 3, "genre1"),
    Venue("v2", 1, 4, "genre2"),
    Venue("v3", 2, 5, "genre3"),
    Venue("v4", 3, 6, "genre4"),
    Venue("v5", 5, 7, "genre1"),
    Venue("v6", 6, 8, "genre2"),
    Venue("v7", 7, 9, "genre3"),
    Venue("v8", 8, 10, "genre4"),
    Venue("v9", 9, 11, "genre1"),
    Venue("v10",10, 12, "genre2"),
]

venues2 = [
    Venue("v1", 1, 3, "genre1"),
    Venue("v2", 4, 6, "genre2"),
    Venue("v3", 5, 7, "genre1"),
    Venue("v4", 7, 9, "genre3"),
    Venue("v5", 9,11, "genre3"),
    Venue("v6",10,12, "genre2"),
]

# -----------------------
# Run search & plot for Test Case 1
# -----------------------
problem1 = TomorrowlandSearch(venues1, genres, festival_end)
path1, expanded1, trace1, edges1, node_states1, parent1, goal_node1 = best_first_tree(problem1, htype=2)
print("Test Case 1 - Path (should be ['v1','v4','v7','v10']):")
print("Path:", path1)
print("Nodes Expanded:", expanded1)
print("Tree Edges (explored only):", len(edges1))
plot_traversal_tree(edges1, node_states1, parent1, goal_node1, "Traversal Tree - Test Case 1 (Only Explored Nodes)")

# -----------------------
# Run search & plot for Test Case 2
# -----------------------
problem2 = TomorrowlandSearch(venues2, genres, festival_end)
path2, expanded2, trace2, edges2, node_states2, parent2, goal_node2 = best_first_tree(problem2, htype=2)
print("\nTest Case 2 - No Path Exists (expected Path: None):")
print("Path:", path2)
print("Nodes Expanded:", expanded2)
print("Tree Edges (explored only):", len(edges2))
plot_traversal_tree(edges2, node_states2, parent2, goal_node2, "Traversal Tree - Test Case 2 (Only Explored Nodes)")

# -----------------------
# Optional: small venue timeline (same as before)
# -----------------------
def plot_venue_map(venues, title):
    plt.figure(figsize=(10,4))
    colors = plt.cm.tab10.colors
    genre_map = {f"genre{i+1}": colors[i] for i in range(10)}
    plotted_labels = set()
    y_positions = {v.id: idx for idx, v in enumerate(venues[::-1], start=1)}  # assign y coords stable
    for v in venues:
        y = y_positions[v.id]
        plt.plot([v.start, v.end], [y, y], linewidth=6, solid_capstyle="butt", label="_nolegend_",
                 color=genre_map[v.genre])
        plt.text((v.start+v.end)/2, y, f"{v.id}\n{v.genre}", ha='center', va='center', color='white', fontsize=8)
        if v.genre not in plotted_labels:
            plt.plot([], [], color=genre_map[v.genre], label=v.genre, linewidth=6)
            plotted_labels.add(v.genre)
    plt.yticks([])
    plt.xlabel("Time")
    plt.title(title)
    plt.legend(ncol=4)
    plt.grid(alpha=0.3)
    plt.show()

plot_venue_map(venues1, "Venue timeline - Test Case 1")
plot_venue_map(venues2, "Venue timeline - Test Case 2")