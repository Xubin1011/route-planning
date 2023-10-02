import pandas as pd
import numpy as np
import networkx as nx  # 用于实现Dijkstra算法
import os

from dijkstra_graph import haversine, x_source, y_source, x_target, y_target

# select the closest node in graph G
def get_closest_node(G, latitude, longitude):
    closest_node = None
    closest_distance = float('inf')

    for node in G.nodes(data=True):
        node_latitude = node[1]['latitude']
        node_longitude = node[1]['longitude']

        # 计算给定节点与目标经纬度坐标之间的距离
        distance = haversine(latitude, longitude, node_latitude, node_longitude)

        if distance < closest_distance:
            closest_node = node[0]
            closest_distance = distance

    return closest_node

# 加载权重矩阵
weights_df = pd.read_csv('dijkstra_edges.csv')
weights_matrix = weights_df.values

# 加载节点坐标数据
pois_df = pd.read_csv('dijkstra_pois.csv')
latitude = pois_df['Latitude'].values
longitude = pois_df['Longitude'].values

# 创建有向图
G = nx.DiGraph()

source = get_closest_node(G, x_source, y_source)
target = get_closest_node(G, x_target, y_target)

print("source:", source)
print("target:", target)

# 添加节点到图中
for i, row in pois_df.iterrows():
    G.add_node(i, latitude=row['Latitude'], longitude=row['Longitude'])

# 添加边和权重到图中
for i in range(len(pois_df)):
    for j in range(len(pois_df)):
        weight = weights_matrix[i, j]
        if weight != np.inf:
            G.add_edge(i, j, weight=weight)

k = 1
shortest_paths = []
for _ in range(k):
    shortest_path = nx.shortest_path(G, source=source, target=target, weight='weight', method="dijkstra")
    shortest_paths.append(shortest_path)
    # for edge in shortest_path:
    #     # 在图中删除已找到的路径，以找到下一条最短路径
    #     G.remove_edge(edge[0], edge[1])

print("Shortest Paths:", shortest_paths)

# 假设shortest_paths包含了最短路径的字典列表
for path_dict in shortest_paths:
    path = list(path_dict.values())[0]  # 获取字典中的路径
    print("Shortest Path:", path)

#
# # 使用Dijkstra算法计算前100条最短路径
# shortest_paths = []
# counter = 0
# for source in range(len(pois_df)):
#     for target in range(len(pois_df)):
#         if source != target:
#             path = nx.shortest_path(G, source=source, target=target, weight='weight')
#             shortest_paths.append(path)
#             counter += 1
#             if counter >= 5:
#                 break
#     if counter >= 5:
#         break
#
# # 创建文件夹以保存路径
# if not os.path.exists('shortest_paths'):
#     os.makedirs('shortest_paths')
#
# # 保存每条路径的节点到不同的CSV文件
# for i, path in enumerate(shortest_paths):
#     path_df = pois_df.loc[path]  # 从节点坐标数据中获取路径上的点
#     path_df.to_csv(f'shortest_paths/path_{i}.csv', index=False)
#
# print("Top 100 shortest paths saved.")
