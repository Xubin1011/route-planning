N, M, K = map(int, input().split())
mushrooms = set()
for _ in range(K):
    x, y = map(int, input().split())
    mushrooms.add((x, y))

# 初始化dp数组，dp[i][j]表示从(1, 1)到达(i, j)不碰到蘑菇的概率
dp = [[0.0] * (M + 1) for _ in range(N + 1)]
print(dp)

# 初始化起点概率为1
dp[1][1] = 1.0

# 从(1, 1)开始正向计算dp数组
for i in range(1, N + 1):
    for j in range(1, M + 1):
        # 如果当前格子有蘑菇，则概率为0
        if (i, j) in mushrooms:
            dp[i][j] = 0.0
        else:
            # 根据题目要求，A只能向右或向上移动
            # 更新dp[i][j]为从上方格子和左方格子过来的概率之和
            dp[i][j] = dp[i - 1][j] * (1 if i > 1 else 1) + dp[i][j - 1] * (1 if j > 1 else 1)

# 输出结果，dp[N][M]表示从(1, 1)到达(n, m)不碰到蘑菇的概率
result = round(dp[N][M], 2)
print(result)
