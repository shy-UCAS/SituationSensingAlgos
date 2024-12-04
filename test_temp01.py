import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def plot_clusters_with_convex_hull(points, labels):
    """
    绘制带有不同颜色和外部包络曲线的聚类划分图。
    
    :param points: 坐标点数组，形状为 (N, 2)，每行是一个点的 (x, y) 坐标。
    :param labels: 每个点的聚类标签数组，长度为 N，值表示点所属的聚类。
    """
    # 获取唯一的聚类标签
    unique_labels = np.unique(labels)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']  # 颜色列表，支持多个簇

    # 创建绘图
    plt.figure(figsize=(8, 6))

    for label in unique_labels:
        # 获取属于当前聚类的点
        cluster_points = points[labels == label]
        
        # 绘制聚类点
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                    label=f"Cluster {label}", color=colors[label], s=100, edgecolor='black', zorder=2)
        
        if len(cluster_points) == 2:
            # 如果聚类点数等于2，绘制连接这两个点的线段
            plt.plot(cluster_points[:, 0], cluster_points[:, 1], 
                     color=colors[label], linewidth=2, zorder=3)
        
        elif len(cluster_points) > 2:
            # 如果聚类点数大于2，绘制外部包络曲线（凸包）
            hull = ConvexHull(cluster_points)
            hull_points = cluster_points[hull.vertices]
            # 绘制凸包曲线
            plt.fill(hull_points[:, 0], hull_points[:, 1], 
                     color=colors[label], alpha=0.3, edgecolor=colors[label], zorder=1)

    # 图形美化
    plt.title("Cluster Visualization with Convex Hulls", fontsize=14)
    plt.xlabel("X Coordinate", fontsize=12)
    plt.ylabel("Y Coordinate", fontsize=12)
    plt.axhline(0, color='black', linewidth=0.5, linestyle="--")  # x轴参考线
    plt.axvline(0, color='black', linewidth=0.5, linestyle="--")  # y轴参考线
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.show()

# 示例坐标点
points = np.array([
    [1, 2],   # Cluster 1
    [2, 3],   # Cluster 1
    [1.5, 1], # Cluster 1
    [6, 5],   # Cluster 2
    [7, 6],   # Cluster 2
    [10, 10]  # Cluster 3 (single point)
])

# 聚类标签：第1~3点属于Cluster 0，第4~5点属于Cluster 1，第6点属于Cluster 2
labels = np.array([0, 0, 0, 1, 1, 2])

# 调用函数绘制
plot_clusters_with_convex_hull(points, labels)