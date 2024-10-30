from pandas import DataFrame
import matplotlib.pyplot as plt
import ast
import numpy as np

log = '/home/tongping/mingcan/github/Pruning/TreeMTL/logs/backup/resnet_1117/resnet_1117.txt'
mIoU = []
Pixel_Acc = []

Angle_Mean = []
Angle_Median = []
Angle_1 = []
Angle_2 = []
Angle_3 = []

abs_err = []
rel_err = []
sigma_1 = []
sigma_2 = []
sigma_3 = []

test_score = []

with open(log, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if 'mIoU' in line:
            dictionary = ast.literal_eval(line.split(': ', 1)[1])
            mIoU.append(dictionary['mIoU'])
            Pixel_Acc.append(dictionary['Pixel Acc'])
        if 'Angle Mean' in line:
            dictionary = ast.literal_eval(line.split(': ', 1)[1])
            Angle_Mean.append(dictionary['Angle Mean'])
            Angle_Median.append(dictionary['Angle Median'])
            Angle_1.append(dictionary['Angle 11.25'])
            Angle_2.append(dictionary['Angle 22.5'])
            Angle_3.append(dictionary['Angle 30'])
        if 'abs_err' in line:
            dictionary = ast.literal_eval(line.split(': ', 1)[1])
            abs_err.append(dictionary['abs_err'])
            rel_err.append(dictionary['rel_err'])
            sigma_1.append(dictionary['sigma_1.25'])
            sigma_2.append(dictionary['sigma_1.25^2'])
            sigma_3.append(dictionary['sigma_1.25^3'])
        if 'test score: ' in line:
            test_score.append(float(line.split('test score:')[1]))
mIoU = [i*100 for i in mIoU]
Pixel_Acc = [i*100 for i in Pixel_Acc]

def save_ecxel():
    df = DataFrame({
                "sparsity": sparsity,
                "mIoU" : mIoU,
                "Pixel_Acc" : Pixel_Acc,
                "Angle_Mean" : Angle_Mean,
                "Angle_Median" : Angle_Median,
                "Angle_1" : Angle_1,
                "Angle_2" : Angle_2,
                'Angle_3' : Angle_3,
                "abs_err" : abs_err,
                "rel_err" : rel_err,
                "sigma_1" : sigma_1,
                "sigma_2" : sigma_2,
                "sigma_3" : sigma_3,
                "test_score" : test_score,
        })
    df.to_excel('logs/test_results.xlsx', sheet_name='sheet1', index=False)


def plot_metrics(backbone="mobilenet", x_ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]):
    if backbone == "mobilenet":
        std_mIoU = 20.36
        std_Pixel_Acc = 49.44
        std_Angle_1 = 28.37

        std_sigma_1 = 47.92
        std_sigma_2 = 78.46
        std_sigma_3 = 92.81
    else:
        std_mIoU = 27.12
        std_Pixel_Acc = 59.13
        std_Angle_1 = 29.4

        std_sigma_1 = 57.91
        std_sigma_2 = 86.61
        std_sigma_3 = 96.37

    # independent model
    # plt.axhline(y=std_mIoU, color='r', linestyle='--')
    # plt.axhline(y=std_Pixel_Acc, color='g', linestyle='--')

    plt.plot(sparsity, mIoU,'s-',color = 'r',label="mIoU")#s-:方形
    plt.plot(sparsity, Pixel_Acc,'o-',color = 'g',label="Pixel_Acc")#o-:圆形
    plt.xlabel("Sparsity")  #横坐标名字
    plt.xticks(x_ticks)    #横坐标刻度
    plt.ylabel("Accuracy")  #纵坐标名字
    plt.title("Segmentation Task")
    plt.legend(loc = "best")#图例
    plt.savefig('logs/segmentation.png')

    plt.cla()
    # plt.axhline(y=std_Angle_1, color='r', linestyle='--')
    plt.plot(sparsity, Angle_1,'s-',color = 'r',label="Pixel_Acc 11.25")#s-:方形
    # plt.plot(sparsity, Angle_2,'o-',color = 'g',label="Pixel_Acc 22.5")#o-:圆形
    # plt.plot(sparsity, Angle_3,'*-',color = 'b',label="Pixel_Acc 30")
    plt.xlabel("Sparsity")  #横坐标名字
    plt.xticks(x_ticks)    #横坐标刻度
    plt.ylabel("Accuracy")  #纵坐标名字
    plt.yticks([10, 20, 30, 40])
    plt.title("Surface Normal prediction")
    plt.legend(loc = "best")#图例
    plt.savefig('logs/Normal1.png')

    plt.cla()
    # plt.axhline(y=std_sigma_1, color='r', linestyle='--')
    # plt.axhline(y=std_sigma_2, color='g', linestyle='--')
    # plt.axhline(y=std_sigma_3, color='b', linestyle='--')

    plt.plot(sparsity, sigma_1,'s-',color = 'r',label="sigma 1.25^1")#s-:方形
    plt.plot(sparsity, sigma_2,'o-',color = 'g',label="sigma 1.25^2")#o-:圆形
    plt.plot(sparsity, sigma_3,'*-',color = 'b',label="sigma 1.25^3")
    plt.xlabel("Sparsity")  #横坐标名字
    plt.xticks(x_ticks)    #横坐标刻度
    plt.ylabel("Accuracy")  #纵坐标名字
    # plt.yticks([10, 20, 30, 40])
    plt.title("Depth estimation ")
    plt.legend(loc = "best")#图例
    plt.savefig('logs/depth.png')

    # test score
    plt.cla()
    plt.plot(sparsity, test_score,'s-',color = 'r',label="test score")#s-:方形
    plt.xlabel("Sparsity")  #横坐标名字
    plt.xticks(x_ticks)    #横坐标刻度
    plt.ylabel("Accuracy")  #纵坐标名字
    plt.yticks(np.arange(0.8, 1.4, 0.05))
    plt.title("Score of normalized 12 metrics")
    plt.legend(loc = "best")#图例
    plt.savefig('logs/final_score.png')


if __name__ == '__main__':
    # sparsity = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.97, 0.98, 0.99, 0.995, 0.999] # mobilenet
    sparsity = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.999] # 15 level
    # sparsity = [0.85, 0.9, 0.93, 0.95, 0.97, 0.99] # 6 level

    # plot_metrics(x_ticks=sparsity)
    # save_ecxel()

    print(test_score)