import matplotlib.pyplot as plt

sparsity = [0.85, 0.9, 0.93, 0.95, 0.97, 0.99] # 6 level
x_ticks = sparsity

# Resnet
# plt.plot(sparsity, [1.2289, 1.2497, 1.2416, 1.2706, 1.2647, 1.1981]
# ,'s-',color = 'r',label="75% backbone sparsity")#s-:方形
# plt.plot(sparsity, [1.2126, 1.2403, 1.257, 1.2391, 1.2708, 1.2036]
# ,'o-',color = 'g',label="90% backbone sparsity")#o-:圆形
# plt.plot(sparsity, [1.2005, 1.1901, 1.2409, 1.2373, 1.2429, 1.1706]
# ,'*-',color = 'b',label="97.2% backbone sparsity")

# Mobilenet
plt.plot(sparsity, [0.9858, 0.9984, 0.9762, 1.0017, 1.0023, 0.9999]
,'s-',color = 'r',label="46.3% backbone sparsity")#s-:方形
plt.plot(sparsity, [1.0002, 0.9866, 0.9918, 0.9959, 1.0027, 1.0156]
,'o-',color = 'g',label="76.2% backbone sparsity")#o-:圆形
plt.plot(sparsity, [0.9945, 0.9715, 0.9958, 0.9868, 0.9946, 0.9894]
,'*-',color = 'b',label="90.0% backbone sparsity")
plt.plot(sparsity, [0.9317, 0.9258, 0.9151, 0.9409, 0.9358, 0.927]
,'x-',color = 'y',label="97.2% backbone sparsity")


plt.xlabel("Task Head Sparsity")  #横坐标名字
plt.xticks(x_ticks)    #横坐标刻度
plt.ylabel("Average Accuracy")  #纵坐标名字
# plt.yticks([10, 20, 30, 40])
# plt.title("Average score on NVUv2 using ResNet")
plt.title("Average score on NVUv2 using MobileNetv2")
plt.legend(loc = "best")#图例
plt.savefig('logs/final_scores.png')
