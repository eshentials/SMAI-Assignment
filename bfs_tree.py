import matplotlib.pyplot as plt
import networkx as nx
from collections import deque

# -----------------------
# BFS with shortest path
# -----------------------
def bfs_tree(problem):
    start = (0, "entrance", set(), ["entrance"])
    q = deque([start])
    visited = set()
    expanded = 0
    trace = []
    edges = []
    node_ids = {"entrance": 0}
    node_count = 1
    shortest_path = None

    while q:
        state = q.popleft()
        expanded += 1
        t, u, covered, path = state
        trace.append((expanded, len(covered)))

        key = (u, tuple(sorted(covered)))
        if key in visited:
            continue
        visited.add(key)

        if problem.goal_test(state) and shortest_path is None:
            shortest_path = path

        for v in problem.venues:
            if v.start >= t and v.end <= problem.festival_end:
                new_covered = covered | {v.genre}
                new_state = (v.end, v.id, new_covered, path + [v.id])

                for node in [u, v.id]:
                    if node not in node_ids:
                        node_ids[node] = node_count
                        node_count += 1

                edges.append((node_ids[u], node_ids[v.id]))
                q.append(new_state)

    return shortest_path, expanded, trace, edges, node_ids

# -----------------------
# Hierarchical Tree Plot
# -----------------------
def plot_tree_hierarchy(edges, node_ids, title, highlight_path=None):
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # Use Graphviz "dot" layout for hierarchy
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    plt.figure(figsize=(12,6))

    # Draw all nodes
    nx.draw(G, pos, with_labels=True, labels={v:k for k,v in node_ids.items()},
            node_color='lightblue', node_size=800, arrows=True, arrowsize=20)

    # Highlight path if provided
    if highlight_path:
        path_edges = [(node_ids[highlight_path[i]], node_ids[highlight_path[i+1]])
                      for i in range(len(highlight_path)-1)]
        nx.draw_networkx_nodes(G, pos, nodelist=[node_ids[n] for n in highlight_path],
                               node_color='orange', node_size=900)
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)

    plt.title(title)
    plt.show()

# -----------------------
# Example Usage
# -----------------------
genres = [f"genre{i}" for i in range(1, 11)]
festival_end = 20

venues1 = [
    Venue("v1", 1, 3, "genre1"),
    Venue("v2", 3, 5, "genre2"),
    Venue("v3", 5, 7, "genre3"),
    Venue("v4", 7, 9, "genre4"),
    Venue("v5", 9, 11, "genre5"),
    Venue("v6", 11, 13, "genre6"),
    Venue("v7", 13, 15, "genre7"),
    Venue("v8", 15, 17, "genre8"),
    Venue("v9", 17, 18, "genre9"),
    Venue("v10", 18, 20, "genre10"),
]

problem1 = TomorrowlandSearch(venues1, genres, festival_end)
shortest_path, expanded, trace, edges, node_ids = bfs_tree(problem1)
print("Shortest Path:", shortest_path)

plot_tree_hierarchy(edges, node_ids, "BFS Tree Hierarchy - Test Case 1", highlight_path=shortest_path)
