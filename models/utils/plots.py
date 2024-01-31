from matplotlib.transforms import Bbox
from sklearn.tree import plot_tree, export_graphviz
import pydotplus
import collections
import re


def full_extent(ax, pad=0.0):
    """
    From: https://stackoverflow.com/questions/4325733/save-a-subplot-in-matplotlib
    Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
    # items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax.get_xaxis().get_label(), ax.get_yaxis().get_label()]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)


def plot_tree_thr(tree, proba_thr, max_depth, features, save_path, trace="CAIDA"):
    dot_data = export_graphviz(
        tree, 
        max_depth=max_depth, 
        feature_names=features,
        filled=True,
        rounded=True,
        class_names=["Mice", "Eleph"],
        proportion=True,
        impurity=False,
    )
    
    graph = pydotplus.graph_from_dot_data(dot_data)
    nodes = graph.get_node_list()
    for i, n in enumerate(nodes):
        if n.get_label() is None: # Artifact from pydotplus
            print(f"Make node {i} disappear")
            n.set_label(trace)
            n.set_fontcolor("black")
            n.set_fillcolor("white")
                
    colors = ('orange', 'lightblue', 'white')
    edges_right = collections.defaultdict(list)
    edges_left = collections.defaultdict(list)

    for edge in graph.get_edge_list():
        if edge.get_source() in edges_right:
            edges_left[edge.get_source()].append(int(edge.get_destination()))
            edges_left[edge.get_source()].append(int(edge.get_source()))
        else:
            edges_right[edge.get_source()].append(int(edge.get_destination()))
            edges_right[edge.get_source()].append(int(edge.get_source()))

    # Color nodes
    for edge in edges_right:
        edges_right[edge].sort()    
        for i in range(2):
            dest = graph.get_node(str(edges_right[edge][i]))[0]
            label = dest.get_label()
            new_label = label
            new_label = re.sub(r'class.*?Mice', '', label)
            new_label = re.sub(r'class.*?Eleph', '', new_label)
            new_label = re.sub(r'value.*?,', 'proba =', new_label)
            new_label = new_label.replace("]", "")
            new_label = new_label.replace("ip ", "ip bit ")
            new_label = new_label.replace("port ", "port bit ")
            new_label = new_label.replace("_6", " (5pk)")
            new_label = new_label.replace("_5", " (5pk)")
            new_label = new_label.replace("_", " ")
            if "<" in dest.get_label():
                dest.set_fillcolor(colors[2])
            else:
                try:
                    proportions = eval(dest.get_label().split("value = ")[1].split("\\")[0])
                except IndexError:
                    proportions = False
                if proportions:
                    dest.set_fillcolor(colors[0] if proportions[1] >= proba_thr else colors[1])
            dest.set_label(new_label)
            
    for edge in edges_left:
        edges_left[edge].sort()    
        for i in range(2):
            dest = graph.get_node(str(edges_left[edge][i]))[0]
            label = dest.get_label()
            new_label = re.sub(r'class.*?Mice', '', label)
            new_label = re.sub(r'class.*?Eleph', '', new_label)
            new_label = re.sub(r'value.*?,', 'proba =', new_label)
            new_label = new_label.replace("]", "")
            new_label = new_label.replace("ip ", "ip bit ")
            new_label = new_label.replace("port ", "port bit ")
            new_label = new_label.replace("_6", " (5pk)")
            new_label = new_label.replace("_5", " (5pk)")
            new_label = new_label.replace("_", " ")
            new_label = new_label.replace("bit bit ", "bit ")
            if "<" in dest.get_label():
                dest.set_fillcolor(colors[2])
            else:
                try:
                    proportions = eval(dest.get_label().split("value = ")[1].split("\\")[0])
                except IndexError:
                    proportions = False
                if proportions:
                    dest.set_fillcolor(colors[0] if proportions[1] >= proba_thr else colors[1])
            dest.set_label(new_label)
            
    # Weight edges
    for edge in edges_right:
        edges_right[edge].sort()
        dest = graph.get_node(str(edges_right[edge][1]))[0]
        try:
            samples = float(dest.get_label().split("samples = ")[1].split("%\\")[0]) / 100 
        except IndexError:
            samples = False
        if samples:
            weight = max(0.4, (samples) * 9)
            current_edge = graph.get_edge(edge, str(edges_right[edge][1]))[0]
            current_edge.set_penwidth(weight)

    for edge in edges_left:
        edges_left[edge].sort()
        dest = graph.get_node(str(edges_left[edge][1]))[0]
        try:
            samples = float(dest.get_label().split("samples = ")[1].split("%\\")[0]) / 100 
        except IndexError:
            samples = False
        if samples:
            weight = max(0.4, (samples) * 9)
            current_edge = graph.get_edge(edge, str(edges_left[edge][1]))[0]
            current_edge.set_penwidth(weight)

    graph.write_pdf(save_path)