
# coding: utf-8

# In[1]:


# Step 1: Importing the required packages
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
from random import randint
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


#Reading the dataset
data = pd.read_csv("Project 1 - Dataset.csv")


# In[3]:


#Defining Constants
N = 10
# No of Columns
P = 10
#total number of weights
Npop = 1000
#Rounding decimals
roundDecimals = 3
#List Containing Fitness Values
parentList = []
#Threshold indictes final iteration where we get our final fitness value
threshold= 50


# In[4]:


def normalize_pd_data(data):
    data_array = data.values
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data_array)
    return pd.DataFrame(data_scaled).round(decimals=roundDecimals)


# In[5]:


# Step 3: Choosing the first 5 columns as input, and the very last column as output (target)
input_data = data.iloc[:,:10]
output_data = data.iloc[:,13:14]


# In[6]:


# Step 4: Creating 25% of the dataset (random) as testing and the rest 75% as training samples
training_input_data = input_data.iloc[0 : int(0.75 * len(input_data)),:]
test_input_data = input_data.iloc[0 : int(0.25 * len(input_data)),:]
training_output_data = output_data.iloc[0 : int(0.75 * len(output_data)),:]
test_output_data = output_data.iloc[0 : int(0.25 * len(output_data)),:]


# In[7]:


# Step 5: Normalise the training dataset with values between 0 and 1.
training_input_data = normalize_pd_data(training_input_data)
training_output_data = normalize_pd_data(training_output_data)
training_output_data_array = np.asarray(training_output_data)

test_input_data = normalize_pd_data(test_input_data)
test_output_data = normalize_pd_data(test_output_data)


# In[8]:


#Step 6 : Calculate the number of parameters (weights) you need to tune in the STRUCTURE (refer to slide #2). You need to tune PxN parameters (weights).
weights = np.matrix(np.random.uniform(low=-1, high= 1, size=(Npop,N*P)),dtype="float64").round(decimals=roundDecimals)
test_weight_matrix = weights.copy()
before = test_weight_matrix[0,0]
weight_array = np.array(weights)


# In[9]:


# Function to calculate exponent
def calculate_exponent(x):
    return round((1/(1+math.exp(-x))),roundDecimals)


# In[10]:


def normalize_martix(data,start,end):
    scaler = MinMaxScaler(copy=False, feature_range=(start, end))
    scaler.fit(data)
    return scaler.transform(data).round(decimals=roundDecimals)


# In[11]:


def calculateProduct(weightMatrixItem,data):
    weight_matrix = weightMatrixItem.reshape(P,N)
    weight_product = np.matrix(data) * weight_matrix
    return np.asarray(weight_product)


# In[12]:


#function to calculate y_hat
def calculateYHat(weight_array):
    y_hat_list = []
    for i in range(0,len(weight_array)):
        y_hat = 0 
        for j in range(0,len(weight_array[i])):
            y_hat = y_hat + calculate_exponent(weight_array[i][j])
        y_hat_list.append(round(y_hat,roundDecimals))
    return y_hat_list


# In[13]:


#Function to calculate fitness
def calculateFitness(yHatList,data):
    numerator = 0
    for i in range(0,len(yHatList)):
        numerator = numerator + (((yHatList[i] - data[i][0])**2))
    ratio = numerator/len(data)
    fitness = (1- ratio) * 100
    return fitness


# In[14]:


#Function to create a chromosome matrix
def createChromosome(weights):
    binary_weight_list = []
    for i in range(0,len(weights)):
        chromosome =''
        for j in range(0,len(weights[i])):
            chromosome+=str(np.binary_repr(int(weights[i][j]), width=10))
        binary_weight_list.append(chromosome)
    return np.matrix(binary_weight_list).reshape(len(weights),1)


# In[15]:


#Function to do crossover
def crossover(parent,weight):
    childArray = []
    for i in range(len(weight)):
        crossover_point = randint(1, len(parent))
        left_parent_bits = parent[0:crossover_point]
        right_parent_bits = parent[crossover_point:len(parent)]
        left_weight_bits = weight[i,0][0:crossover_point]
        right_weight_bits = weight[i,0][crossover_point:len(parent)]
        childArray.append(left_parent_bits + right_weight_bits)
        childArray.append(left_weight_bits + right_parent_bits)
    return np.matrix(childArray).reshape(len(childArray),1)


# In[16]:


def mutation(crossoverMatrix):
    for i in range(len(crossoverMatrix)):
        flipOverBitsSize = int(0.05 * len(crossoverMatrix[i,0]))
        mutationPoints = random.sample(range(1, len(crossoverMatrix[i,0])), flipOverBitsSize)
        crossoverItem = list(crossoverMatrix[i,0])
        for j in range(len(mutationPoints)):
            bit = crossoverItem[j]
            if(bit == "1"):
                bit = "0"
            else:
                bit = "1"
            crossoverItem[j] = bit
        crossoverMatrix[i,0] = "".join(crossoverItem)
    return crossoverMatrix


# In[17]:


def binarize(weights):
    normalized_matrix = normalize_martix(weights,0,1).astype(float)
    normalize_weights = (normalized_matrix * 1000).astype(int)
    return createChromosome(np.asarray(normalize_weights))


# In[18]:


def debinarize(mutationMatrix):
    debinarizeList =[]
    for i in range(len(mutationMatrix)):
        row = []
        for j in range(N*P): 
            start = j*10
            end = j*10+10
            row.append(int(mutationMatrix[i,0][start:end],2)/1000)
        debinarizeList.append(row)
    return normalize_martix(np.matrix(debinarizeList),-1,1)


# In[19]:


#Get The parent
def getParent(weightMatrix,data):
    fitnessList = []
    for i in range(0,len(weightMatrix)):
        productArray = calculateProduct(weightMatrix[i],data)
        yHatList = calculateYHat(productArray)
        fitness = calculateFitness(yHatList,training_output_data_array)
        fitnessItem = ((round(fitness,roundDecimals)),i)
        fitnessList.append(fitnessItem)
    return fitnessList


# In[20]:


def createParent(previousParent,previousWeightMatrix,weightMatrix,data):
    fitnessList = getParent(weightMatrix,data)
    fitnessList = sorted(fitnessList, key=lambda x: x[0],reverse=True)
    currentMaxFitness = fitnessList[0][0]
    if(len(parentList) > 0):
        if(currentMaxFitness > parentList[len(parentList)-1]):
            #Creating new Weight Matrix
            newWeightArray =[]
            for i in range(0,Npop):
                newWeightArray.append(weightMatrix[fitnessList[i][1]])
            binary_weight_matrix = binarize(np.matrix(newWeightArray).reshape(Npop,N*P))
            parent = binary_weight_matrix[0,0]
        else:
            binary_weight_matrix = previousWeightMatrix
            parent = previousParent
    else:
        parentWeights = weightMatrix[fitnessList[0][1]]
        binary_weight_matrix = binarize(weightMatrix)
        parentMatrix = binarize(np.matrix(parentWeights).reshape(2,int((N*P)/2)))
        parent = parentMatrix[0,0] + parentMatrix[1,0]
    parentList.append(currentMaxFitness)
    return (parent,binary_weight_matrix)


# In[21]:


def main():
    iteration = 1
    initialParent = "0"
    parent = createParent(initialParent,weights,weights,training_input_data)
    while(iteration < threshold):
        iteration = iteration + 1
        crossoverMatrix = crossover(parent[0],parent[1])
        mutationMatrix = mutation(crossoverMatrix)
        debinarizeMatrix = debinarize(mutationMatrix)
        parent = createParent(parent[0],parent[1],debinarizeMatrix,training_input_data)
    return parent[1]


# In[22]:


optimizedWeightMatrix = main()


# In[23]:


debinarizeWeightMatrix = debinarize(optimizedWeightMatrix)


# In[24]:


#print(parentList)


# In[25]:


#scatter plot of fitness value against iteration
print("Scatter plot of Highest Fitness Value of each iteration")
plt.scatter(list(range(0, len(parentList))), parentList)
plt.show()


# In[26]:


#Calculating y_hat for test data
y_hatList_test =[]
for i in range(0,Npop):
    productArray = calculateProduct(debinarizeWeightMatrix[i],test_input_data)
    y_hatList_test = calculateYHat(productArray)


# In[27]:


#3D Scatter plot
print("3D scatter plot of y_hat_test and y_test")
fig = plt.figure()
axis = Axes3D(fig)

weight_X = list(np.asarray(test_input_data.iloc[:,0]))
height_Y = list(np.asarray(test_input_data.iloc[:,1]))
y_test = list(np.asarray(test_output_data.iloc[:,0]))

axis.scatter(weight_X, height_Y, y_test,c='g')
axis.scatter(weight_X, height_Y, y_hatList_test,c='r')
plt.show()


# In[28]:


#Calculating Error 
numerator = 0
for i in range(len(y_test)):
    numerator = numerator + ((y_hatList_test[i]-y_test[i])**2)
error = round(numerator/len(y_test),1)
print("Error:",error)

