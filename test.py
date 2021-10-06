from scipy.io import loadmat


A = loadmat('labelTest/SVMlabelTest.mat')  # len = 330
print(A)
B = loadmat('labelTest/SOMLabelTest.mat')  # len = 330
print(B)
C = loadmat('labelTest/RBFLabelTest.mat')  # len = 330
print(C)
D = loadmat('labelTest/KMeansLabelTest.mat')  # len = 330
print(D)




