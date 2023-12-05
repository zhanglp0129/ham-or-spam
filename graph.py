import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch

loss_list = [0.7043314079443613, 0.6660837928454081, 0.6404755512873331, 0.5938608845074972, 0.5499296287695566, 0.4950043857097626, 0.4171350747346878, 0.3401318738857905, 0.2833571632703145, 0.25963781277338666, 0.19225067645311356, 0.1877980331579844, 0.1897540638844172, 0.22080451746781668, 0.22293669233719507, 0.1659609961012999, 0.15771904091040292, 0.1554302747050921, 0.1693084053695202, 0.13039497969051203, 0.15089872355262438, 0.16303928196430206, 0.15229275388022265, 0.1331207137554884, 0.11409648818274339, 0.10662153859933217, 0.11008928281565507, 0.11280493810772896, 0.0944670041402181, 0.09377905850609143, 0.10174669076999028, 0.10407191763321559, 0.09552135256429513, 0.133237116659681, 0.1153474614645044, 0.11222515938182671, 0.11105409854402144, 0.11470257490873337, 0.12078556294242541, 0.12633209116756916, 0.12133964026967685, 0.09976774702469508, 0.09133249831696351, 0.09478603303432465, 0.08710184196631114, 0.07720072225977977, 0.07989697686086099, 0.08008043033381303, 0.08090174446503322, 0.08974214488019545, 0.08832686394453049, 0.08082446036860347, 0.08182639939089616, 0.08331048876668017, 0.20670978911221027, 0.09677285701036453, 0.08099469464893143, 0.1064019196977218, 0.09702830264965694, 0.09017408949633439, 0.08794176268080871, 0.09524059233566125, 0.11413331422954798, 0.08787716304262479, 0.10530486019949119, 0.12344155025978883, 0.08814322079221408, 0.10059249711533387, 0.16134836648901305, 0.12828833609819412, 0.08550261240452528, 0.07682385047276814, 0.07467590148250262, 0.07828713906928897, 0.0854290012891094, 0.08425731483536462, 0.0828633327037096, 0.08101224876008928, 0.08721423397461574, 0.09786091806987922, 0.1076083288838466, 0.08507537057933708, 0.07699501141905785, 0.07945439033210278, 0.07833704041937987, 0.07938185955087344, 0.08282768074423075, 0.07898908690549433, 0.09053418971598148, 0.0913368829836448, 0.09167964135607083, 0.09122009171793859, 0.09241379991484185, 0.08916751217717926, 0.10175976537478466, 0.24497231530646482, 0.2492716945707798, 0.37543281229833764, 0.2528626731752108, 0.22860521326462427, 0.1103652990811194, 0.10759722627699375, 0.10516596833864848, 0.09996986598707736, 0.10573438446347912, 0.10739303236672033, 0.10592663602437824, 0.1046995980044206, 0.11490051510433356, 0.22705635180075964, 0.35161131372054416, 0.35735112490753335, 0.1477168301741282, 0.15742605303724608, 0.5257390265663465, 0.2249051065494617, 0.10430410271510482, 0.1101373415440321, 0.11114164379735787, 0.1272941374530395]
acc_list = [0.12765330188679244, 0.8655169025157233, 0.8686124213836478, 0.8802820361635221, 0.9056112421383649, 0.9288276336477987, 0.9230788128930817, 0.9415782232704403, 0.9546481918238993, 0.9601267688679246, 0.9647946147798742, 0.9577437106918238, 0.9624606918238993, 0.9718455188679246, 0.9710839229559749, 0.9697818396226415, 0.968971108490566, 0.9736389544025158, 0.9734178459119497, 0.9739583333333334, 0.9603970125786163, 0.9538374606918238, 0.972877358490566, 0.9731476022012578, 0.9760220125786163, 0.9791175314465409, 0.9778154481132075, 0.9775943396226415, 0.9770538522012578, 0.9731476022012578, 0.9700029481132075, 0.9705434355345912, 0.9778645833333334, 0.974719929245283, 0.9796580188679246, 0.9828026729559749, 0.9801985062893083, 0.9783559355345912, 0.9812303459119497, 0.9793877751572326, 0.9788964229559749, 0.9799282625786163, 0.9799282625786163, 0.980689858490566, 0.9817708333333334, 0.9830729166666666, 0.9812303459119497, 0.9828026729559749, 0.9854068396226415, 0.9838345125786163, 0.9841047562893083, 0.9869791666666666, 0.9867089229559749, 0.9856770833333334, 0.9788964229559749, 0.9812303459119497, 0.9817708333333334, 0.9762922562893083, 0.9757517688679246, 0.9770538522012578, 0.9757517688679246, 0.9809601022012578, 0.9749901729559749, 0.9804196147798742, 0.9718455188679246, 0.9679392688679246, 0.9718455188679246, 0.9731476022012578, 0.9598565251572326, 0.9687008647798742, 0.9775943396226415, 0.9801493710691824, 0.9799282625786163, 0.9778645833333334, 0.976783608490566, 0.9775943396226415, 0.978626179245283, 0.98046875, 0.9801985062893083, 0.9812303459119497, 0.9812303459119497, 0.98046875, 0.9815005896226415, 0.9799282625786163, 0.9801985062893083, 0.9812303459119497, 0.9809601022012578, 0.9796580188679246, 0.982532429245283, 0.9828026729559749, 0.9828026729559749, 0.9828026729559749, 0.9841047562893083, 0.9830729166666666, 0.9815005896226415, 0.9796580188679246, 0.9801985062893083, 0.9799282625786163, 0.9817708333333334, 0.9804196147798742, 0.9796580188679246, 0.9809601022012578, 0.9812303459119497, 0.9817708333333334, 0.980689858490566, 0.9812303459119497, 0.9830729166666666, 0.9815005896226415, 0.9809601022012578, 0.9809601022012578, 0.9801985062893083, 0.978626179245283, 0.976783608490566, 0.978626179245283, 0.9770538522012578, 0.9788964229559749, 0.9788964229559749, 0.9809601022012578, 0.9718455188679246, 0.9679392688679246]

def plot_loss(loss_list:list[float]):
    size = len(loss_list)
    x = np.arange(1,size+1)
    y = np.array(loss_list)
    plt.plot(x,y,'r-')
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.show()

def plot_acc(acc_list:list[float]):
    size = len(acc_list)
    x = np.arange(1, size + 1)
    y = np.array(acc_list)
    plt.plot(x, y, 'b-')
    plt.xlabel("iteration")
    plt.ylabel("accuracy")
    plt.show()

def plot_sigmoid():
    x = np.linspace(-10,10,1000)
    y = F.sigmoid(torch.from_numpy(x))
    plt.plot(x, y, 'g-')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

if __name__ == "__main__":
    plot_loss(loss_list)
    plot_acc(acc_list)
    plot_sigmoid()