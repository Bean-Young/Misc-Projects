#test1
print("Hello,world!")
#test2
import numpy as np
import pandas
import scipy
import matplotlib
import PIL
import openpyxl
#test3
arr1 = np.array([[1, 2, 3],
                 [4, 5, 6]])
arr2 = np.array([[1, 2],
                 [3, 4],
                 [5, 6]])
arr_result1 = np.matmul(arr1, arr2)
arr_result2=np.dot(arr1,arr2)
arr_result3=arr1@arr2
#区别见小结
print(arr_result1)
print(arr_result2)
print(arr_result3)
