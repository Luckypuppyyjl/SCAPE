import matplotlib.pyplot as plt

x = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210]
y1 = [84.02, 86.13,86.49,87.31,86.66,86.94,87.50,87.96,87.68,87.59,88.07,87.43,87.42,87.83,87.56,87.66,87.75,88.09,88.43,88.32,88.21]
y2 = [85.80, 87.38,89.00,88.95,88.80,90.29,90.92,90.82,90.86,89.88,90.16,90.21,89.88,90.22,90.01,90.15,90.33,90.09,90.25,90.21,90.23]

plt.title('Epoch/PCK')  # 折线图标题
#plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
plt.xlabel('Epoch')  # x轴标题
plt.ylabel('PCK')  # y轴标题
plt.plot(x, y1, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
plt.plot(x, y2, marker='o', markersize=3)


for a, b in zip(x, y1):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=6)  # 设置数据标签位置及大小
for a, b in zip(x, y2):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=6)


plt.legend(['SCAPE', 'SCAPE with refiner'])  # 设置折线名称

plt.show()  # 显示折线图

