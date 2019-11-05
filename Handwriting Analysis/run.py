
# coding: utf-8

# In[38]:


from sklearn.cluster import KMeans      # Class sklear.cluster enables using KMeans for clustering the dataset
import numpy as np                      # For the use of number functions like array,arange etc.
import csv                              # Enables processing the CSV(comma-separated values) files
import math                             # Enables the use of mathematical operations like exponentials, powers,squareroot etc.
import matplotlib.pyplot                # Enables plotting graphs 
import pandas as pd
from matplotlib import pyplot as plt


## PREPARING RAW DATAGETS
def GenerateRawData(humanmatrix):
    humanmatrix.drop(['img_id_A','img_id_B','target'],axis=1,inplace=True)
    RawData = humanmatrix.values
    RawData = np.transpose(RawData)
    return RawData



# Computes the Spread Of the Radial Basis Functions i.e the variance
def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic):
    BigSigma    = np.zeros((len(Data),len(Data)))                 # Computing a matrix of 9x9 with entries as zero as the length of Data is 9; corresponds to the number of rows
    DataT       = np.transpose(Data)                              # Computing the transpose of the Data matrix;
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))    # Computing the Length of the training data set which is 235058   
    TrainingLen=TrainingLen-1
    varVect     = []                                              # Initializing an array to store the variance
    for i in range(0,len(DataT[0])):                              # running the loop from 0 to 9
        vct = []
        for j in range(0,int(TrainingLen)):                       # running an inner loop from 0 to 235058
            vct.append(Data[i][j])                                # append the values in Date[i][j] to the vct array
        varVect.append(np.var(vct))                               # Compute the variance for the features
        
    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]                               # Appending the computed values of variance along the diagonal of the Covariance matrix
    if IsSynthetic == True:
        BigSigma = np.dot(3,BigSigma)
    else:
        BigSigma = np.dot(200,BigSigma)
    ##print ("BigSigma Generated..")
    return BigSigma

# Calculate the Value of the terms In the powers of the exponential term of the guassian RBF
def GetScalar(DataRow,MuRow, BigSigInv):  
    R = np.subtract(DataRow,MuRow)          # Subtract the values of inputs and the mean and store in R
    T = np.dot(BigSigInv,np.transpose(R))   # Multiply the transpose of R with the Covariance matrix(BigSigma) and store in T
    L = np.dot(R,T)                         # Dot product of R and T gives a scalar value
    return L

# Calculate the Gaussian radial basis function
def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))   # Calculate the gaussian RBF by the formula
    return phi_x

# Generate the design matrix PHI that contains the basis functions for all input features
def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)                                  # Tranpose of the Data matrix; dimensions are now (293823,9)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))  # Length of the training set which is 235058    
    TrainingLen=TrainingLen-1
    PHI = np.zeros((int(TrainingLen),len(MuMatrix)))            # Initialize a Matrix (80% data)xM with entries as zeroes i.e (235058,15)
    BigSigInv = np.linalg.inv(BigSigma)                         # Inverse of the BigSigma matrix 
    for  C in range(0,len(MuMatrix)):                           # running a loop from 0 to 15
        for R in range(0,int(TrainingLen)):                     # running an inner loop from 0 to 235058
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)  # Calculate the RBF value using the formula
    #print ("PHI Generated..")
    return PHI

# Compute the weights of the Closed form solution to minimize the error
def GetWeightsClosedForm(PHI, T, Lambda):
    Lambda_I = np.identity(len(PHI[0]))      # Create an indentity matrix with dimenions as (15,15)     
    # Computing the regularization term of the Closed form equation i.e Lambda multipled with the identity matrix
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda                   # Gives Lambda along the diagonal
    # Compute the Closed form solution equation 
    PHI_T       = np.transpose(PHI)               # Transpose of the PHI matrix i.e. dimensions are (15, 235058)
    PHI_SQR     = np.dot(PHI_T,PHI)               # Dot product of the Phi matrix with its Transpose. Dimensions are (15,15)
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR)        # Add the product with the Regularization term. Dimensions are (15,15)
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)       # Inverse of the sum 
    INTER       = np.dot(PHI_SQR_INV, PHI_T)      # Resultant matrix is multipled with the transpose of PHI. Dimensions are (15, 235058)
    W           = np.dot(INTER, T)                # Finally multipled with the target values of the training set giving a (15,1) shape
    ##print ("Training Weights Generated..")
    return W

# Calulate the Target values of the Testing dataset using the calculated weights
def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI))     # Compute the target values from the product of the adjusted weights and PHI
    ##print ("Test Out Generated..")
    return Y

# Calculate the root mean square value for the Validation data output with respect to the actual validation output
def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    # Compute Sum of the square of differences between the Predicted and Actual data
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2) 
    # Increment counter if the predicted value is equal to the actual value
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):    # np.around() rounds the number to the given number of decimals
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))      # Compute accuarcy of the validation set
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))  # Compute the RMS by finding the squareroot of the mean i.e sum/N




def getweight(RawData,TrainDataSub,TrainTargetSub,TestDataSub,ValidateDataSub):
    M = 5
    # Cluster the Training dataset into M clusters using KMeans
    #for i in M:
    kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainDataSub))
    # Form the Mu matrix with the centers of the M clusters 
    Mu = kmeans.cluster_centers_ 

    BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent,IsSynthetic)    # Compute Spread of Basis functions i.e. covariance matrix
    TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)          # Compute the desgin matrix for training set
    W            = GetWeightsClosedForm(TRAINING_PHI,TrainTargetSub,(C_Lambda))  # Compute weights
    TEST_PHI     = GetPhiMatrix(TestDataSub, Mu, BigSigma, 100) 
    VAL_PHI      = GetPhiMatrix(ValidateDataSub, Mu, BigSigma, 100)
    return W, TRAINING_PHI
   

    #print(Mu.shape)
    #print(BigSigma.shape)
    #print(TRAINING_PHI.shape)
    #print(W.shape)


# In[57]:


def calculateLinR(W,TRAINING_PHI,TrainTargetSub,ValidateTargetSub,TestTargetSub):
    W_Now        = np.dot(220, W)
    La           = 5
    learningRate = 0.02
    L_Erms_Val   = []
    L_Erms_TR    = []
    L_Erms_Test  = []
    W_Mat        = []
    Erms = []
    Acc_Test=[]
    Acc_Train=[]
    Acc_Val=[]
    acc = []
    #for j in learningRate:

    for i in range(0,400):

    #print ('---------Iteration: ' + str(i) + '--------------')
        Delta_E_D     = -np.dot((TrainTargetSub[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i]) 
        La_Delta_E_W  = np.dot(La,W_Now)
        Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
        Delta_W       = -np.dot(learningRate,Delta_E)  # Negative value of the dot product of learning rate and Delta-E
        W_T_Next      = W_Now + Delta_W                # Sum of Initial weights and Delta-W
        W_Now         = W_T_Next                       # Calculate the updated W

        #-----------------TrainingData Accuracy---------------------#
        TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next)                    # Get training target
        Erms_TR       = GetErms(TR_TEST_OUT,TrainTargetSub)                  # Get training E-RMS
        L_Erms_TR.append(float(Erms_TR.split(',')[1]))
        Acc_Train.append(float(Erms_TR.split(',')[0]))

        #-----------------ValidationData Accuracy---------------------#
        #VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next)                         # Get validation target
        #Erms_Val      = GetErms(VAL_TEST_OUT,ValidateTargetSub)              # Get Validation E-RMS
        #L_Erms_Val.append(float(Erms_Val.split(',')[1]))
        #Acc_Val.append(float(Erms_Val.split(',')[0]))

        #-----------------TestingData Accuracy---------------------#
        #TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
        #Erms_Test = GetErms(TEST_OUT,TestTargetSub)                       # Get Tssting target
        #L_Erms_Test.append(float(Erms_Test.split(',')[1]))                   # Get testing E-RMS
        #Acc_Test.append(float(Erms_Test.split(',')[0]))

        #Erms.append(min(L_Erms_Test))
        #acc.append(np.around(max(Acc_Test),5))
        #print(min(L_Erms_Test))
        #print(max(Acc_Test))

    #print(Erms)
    #print(acc)
    return L_Erms_TR, Acc_Train



def printfun(Erms_Test,ACC_Test):
    print ('----------Gradient Descent Solution--------------------')
    print ('----------LINEAR REGRESSION:Feature Concatenation--------------------')
    #print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))           # Print Erms for training set
    #print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))          # Print Erms for validation set
    #print ("Training Accuracy   = " + str(np.around(max(Acc_Train),5)))           # Print Erms for training set
    #print ("Validation Accuracy= " + str(np.around(max(Acc_Val),5)))          # Print Erms for validation set
    print ("Testing Accuracy= " + str(np.around(max(ACC_Test),5)))   
    print ("E_rms Testing    = " + str(np.around(min(Erms_Test),5)))         # Print Erms for testing set


# In[60]:

def printfun1(Erms_Test1,ACC_Test1):
    print ('----------Gradient Descent Solution--------------------')
    print ('----------LINEAR REGRESSION:Feature Subtraction--------------------')
    #print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))           # Print Erms for training set
    #print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))          # Print Erms for validation set
    #print ("Training Accuracy   = " + str(np.around(max(Acc_Train),5)))           # Print Erms for training set
    #print ("Validation Accuracy= " + str(np.around(max(Acc_Val),5)))          # Print Erms for validation set
    print ("Testing Accuracy= " + str(np.around(max(ACC_Test1),5)))   
    print ("E_rms Testing    = " + str(np.around(min(Erms_Test1),5)))         # Print Erms for testing set


# ## LOGISTIC REGRESSION

# In[74]:


def calculateLR(TrainDataConcat,ValidateDataConcat,TestDataConcat,TrainTargetConcat,ValidateTargetConcat,TestTargetConcat):
    DataTrain = np.transpose(TrainDataConcat)
    DataVal   = np.transpose(ValidateDataConcat)
    DataTest  = np.transpose(TestDataConcat)
    W_Now       = np.ones(len(DataTrain[0]))
    W_Now       =np.transpose(W_Now)
    La           = 5
    learningRate = 0.02
    L_Erms_Val   = []
    L_Erms_TR    = []
    L_Erms_Test  = []
    W_Mat        = []
    Erms = []
    Acc_Test=[]
    Acc_Train=[]
    Acc_Val=[]
    acc = []

    def ComputefunctionG(W_Now,phi):
        z=np.dot(np.transpose(W_Now),phi)
        g=math.exp(-z)
        h_x= 1/(1+g)
        return h_x

    #for j in learningRate:

    for i in range(0,400):

    #print ('---------Iteration: ' + str(i) + '--------------')
        G=ComputefunctionG(W_Now,DataTrain[i])
        Delta_E_D     = -np.dot((TrainTargetConcat[i] - G),DataTrain[i]) 
        La_Delta_E_W  = np.dot(La,W_Now)
        Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
        Delta_W       = -np.dot(learningRate,Delta_E)  # Negative value of the dot product of learning rate and Delta-E
        W_T_Next      = W_Now + Delta_W                # Sum of Initial weights and Delta-W
        W_Now         = W_T_Next                       # Calculate the updated W

        #-----------------TrainingData Accuracy---------------------#
        TR_TEST_OUT   = GetValTest(DataTrain,W_T_Next)                    # Get training target
        Erms_TR       = GetErms(TR_TEST_OUT,TrainTargetConcat)                  # Get training E-RMS
        L_Erms_TR.append(float(Erms_TR.split(',')[1]))
        Acc_Train.append(float(Erms_TR.split(',')[0]))

        #-----------------ValidationData Accuracy---------------------#
        VAL_TEST_OUT  = GetValTest(DataVal,W_T_Next)                         # Get validation target
        Erms_Val      = GetErms(VAL_TEST_OUT,ValidateTargetConcat)                     # Get Validation E-RMS
        L_Erms_Val.append(float(Erms_Val.split(',')[1]))
        Acc_Val.append(float(Erms_Val.split(',')[0]))

        #-----------------TestingData Accuracy---------------------#
        TEST_OUT      = GetValTest(DataTest,W_T_Next) 
        Erms_Test = GetErms(TEST_OUT,TestTargetConcat)                            # Get Tssting target
        L_Erms_Test.append(float(Erms_Test.split(',')[1]))                   # Get testing E-RMS
        Acc_Test.append(float(Erms_Test.split(',')[0]))

        #Erms.append(min(L_Erms_Test))
        #acc.append(np.around(max(Acc_Test),5))
        #print(min(L_Erms_Test))
        #print(max(Acc_Test))

    #print(Erms)
    #print(acc)

    return L_Erms_Test, Acc_Test


# In[75]:

def lmain():
    
    TrainingPercent = 80                    # Given data set is partitioned; 80% of the dataset is assigned for training 
    ValidationPercent = 10                  # Given data set is partitioned; 10% of the dataset is assigned for validation
    TestPercent = 10                        # Given data set is partitioned; 10% of the dataset is assigned for testing
    maxAcc = 0.0
    maxIter = 0
    C_Lambda = 0.005                         # Coefficient of the Weight decay regularizer term 
    M = 10                                    # No of Basis functions
    PHI = []
    IsSynthetic = False


    # In[41]:


    ##READING THE CSV FILES AND GENERATING THE DATASETS
    df1= pd.read_csv('same_pairs.csv')
    df2= pd.read_csv('HumanObserved-Features-Data.csv')
    df3= pd.read_csv('diffn_pairs.csv')


    # ## HUMAN OBSERVED DATA- FEATURE CONTATENATION

    # In[42]:


    #merge same_pairs.csv and HumanObserved.csv files
    df4=pd.merge(df1,df2,left_on="img_id_A",right_on="img_id")
    df5=pd.merge(df4,df2,left_on="img_id_B",right_on="img_id")
    df5.drop(['Unnamed: 0_x','Unnamed: 0_y','img_id_x','img_id_y'],axis=1,inplace=True)
    #df5
    #merge diffn_pairs.csv and HumanObserved.csv files
    df6=pd.merge(df3,df2,left_on="img_id_A",right_on="img_id")
    df7=pd.merge(df6,df2,left_on="img_id_B",right_on="img_id")
    df7.drop(['Unnamed: 0_x','Unnamed: 0_y','img_id_x','img_id_y'],axis=1,inplace=True)
    #df7
    human_con=df5.append(df7)
    human_con1=human_con
    #human_con


    # ## HUMAN OBSERVED DATA- FEATURE SUBTRACTION

    # In[43]:


    human_sub=df5.append(df7)
    human_sub["f1"]=human_sub["f1_x"]-human_sub["f1_y"]
    human_sub["f1"]= human_sub["f1"].abs()
    human_sub["f2"]=human_sub["f2_x"]-human_sub["f2_y"]
    human_sub["f2"]= human_sub["f2"].abs()
    human_sub["f3"]=human_sub["f3_x"]-human_sub["f3_y"]
    human_sub["f3"]= human_sub["f3"].abs()
    human_sub["f4"]=human_sub["f4_x"]-human_sub["f4_y"]
    human_sub["f4"]= human_sub["f4"].abs()
    human_sub["f5"]=human_sub["f5_x"]-human_sub["f5_y"]
    human_sub["f5"]= human_sub["f5"].abs()
    human_sub["f6"]=human_sub["f6_x"]-human_sub["f6_y"]
    human_sub["f6"]= human_sub["f6"].abs()
    human_sub["f7"]=human_sub["f7_x"]-human_sub["f7_y"]
    human_sub["f7"]= human_sub["f7"].abs()
    human_sub["f8"]=human_sub["f8_x"]-human_sub["f8_y"]
    human_sub["f8"]= human_sub["f8"].abs()
    human_sub["f9"]=human_sub["f9_x"]-human_sub["f9_y"]
    human_sub["f9"]= human_sub["f9"].abs()
    human_sub.drop(['f1_x','f2_x','f3_x','f4_x','f5_x','f6_x','f7_x','f8_x','f9_x'],axis=1,inplace=True)
    human_sub.drop(['f1_y','f2_y','f3_y','f4_y','f5_y','f6_y','f7_y','f8_y','f9_y'],axis=1,inplace=True)
    human_sub= human_sub[['img_id_A','img_id_B','f1','f2','f3','f4','f5','f6','f7','f8','f9','target']]
    human_sub1 = human_sub
    #human_sub

    ## DATA PARTIONING INTO TRAINING, TESTING AND VALIDATION DATASETS


    # ## DATA PARTITION FOR FEATURE CONTATENATION- HUMAN_OBSERVED

    # In[45]:


    trainingdata_concat = human_con.sample(frac = 0.8)
    trainingdata_concat.drop(['img_id_A','img_id_B'],axis=1,inplace=True)
    testingdata_concat = human_con.drop(trainingdata_concat.index).sample(frac = 0.5)
    validatingdata_concat = human_con.drop(trainingdata_concat.index).drop(testingdata_concat.index)
    validatingdata_concat.drop(['img_id_A','img_id_B'],axis=1,inplace=True)
    testingdata_concat.drop(['img_id_A','img_id_B'],axis=1,inplace=True)


    # ## DATA PARTITION FOR FEATURE SUBTRACTION- HUMAN_OBSERVED

    # In[46]:


    trainingdata_sub= human_sub.sample(frac = 0.8)
    trainingdata_sub.drop(['img_id_A','img_id_B'],axis=1,inplace=True)
    testingdata_sub = human_sub.drop(trainingdata_sub.index).sample(frac = 0.5)
    validatingdata_sub = human_sub.drop(trainingdata_sub.index).drop(testingdata_sub.index)
    validatingdata_sub.drop(['img_id_A','img_id_B'],axis=1,inplace=True)
    testingdata_sub.drop(['img_id_A','img_id_B'],axis=1,inplace=True)


    # # CONVERTING THE DATAFRAMES INTO MATRICES FOR DATA PROCESSING

    # In[47]:



    TrainTargetConcat= trainingdata_concat['target'].values
    TrainTargetConcat = np.transpose(TrainTargetConcat)
    trainingdata_concat.drop(['target'],axis=1,inplace=True)
    TrainDataConcat= trainingdata_concat.values
    TrainDataConcat = np.transpose(TrainDataConcat)

    print(TrainTargetConcat.shape)
    print(TrainDataConcat.shape)


    # In[48]:


    ValidateTargetConcat= validatingdata_concat['target'].values
    ValidateTargetConcat = np.transpose(ValidateTargetConcat)
    validatingdata_concat.drop(['target'],axis=1,inplace=True)
    ValidateDataConcat= validatingdata_concat.values
    ValidateDataConcat = np.transpose(ValidateDataConcat)

    print(ValidateTargetConcat.shape)
    print(ValidateDataConcat.shape)


    # In[49]:


    TestTargetConcat= testingdata_concat['target'].values
    TestTargetConcat = np.transpose(TestTargetConcat)
    testingdata_concat.drop(['target'],axis=1,inplace=True)
    TestDataConcat= testingdata_concat.values
    TestDataConcat = np.transpose(TestDataConcat)

    print(TestTargetConcat.shape)
    print(TestDataConcat.shape)


    # In[50]:


    # CONVERTING THE DATAFRAMES INTO MATRICES FOR DATA PROCESSING
    TrainTargetSub= trainingdata_sub['target'].values
    TrainTargetSub = np.transpose(TrainTargetSub)
    trainingdata_sub.drop(['target'],axis=1,inplace=True)
    TrainDataSub= trainingdata_sub.values
    TrainDataSub = np.transpose(TrainDataSub)

    print(TrainTargetSub.shape)
    print(TrainDataSub.shape)


    # In[51]:


    ValidateTargetSub= validatingdata_sub['target'].values
    ValidateTargetSub = np.transpose(ValidateTargetSub)
    validatingdata_sub.drop(['target'],axis=1,inplace=True)
    ValidateDataSub= validatingdata_sub.values
    ValidateDataSub = np.transpose(ValidateDataSub)

    print(ValidateTargetSub.shape)
    print(ValidateDataSub.shape)


    # In[52]:


    TestTargetSub= testingdata_sub['target'].values
    TestTargetSub = np.transpose(TestTargetSub)
    testingdata_sub.drop(['target'],axis=1,inplace=True)
    TestDataSub= testingdata_sub.values
    TestDataSub = np.transpose(TestDataSub)

    print(TestTargetSub.shape)
    print(TestDataSub.shape)

    RawData = GenerateRawData(human_con1)
    RawData1 = GenerateRawData(human_sub1)
    weight,phi=getweight(RawData,TrainDataConcat,TrainTargetConcat,TestDataConcat,ValidateDataConcat)
    weight1,phi1=getweight(RawData1,TrainDataSub,TrainTargetSub,TestDataSub,ValidateDataSub)
    Erms_Test,ACC_Test= calculateLinR(weight,phi,TrainTargetConcat,ValidateTargetConcat,TestTargetConcat)
    printfun(Erms_Test,ACC_Test)
    Erms_Test1,ACC_Test1= calculateLinR(weight1,phi1,TrainTargetSub,ValidateTargetSub,TestTargetSub)
    printfun1(Erms_Test1,ACC_Test1)
    L_Erms_Test,Acc_Test= calculateLR(TrainDataConcat,ValidateDataConcat,TestDataConcat,TrainTargetConcat,ValidateTargetConcat,TestTargetConcat)
    printfun2(L_Erms_Test,Acc_Test)
    L_Erms_Test1,Acc_Test1= calculateLR(TrainDataSub,ValidateDataSub,TestDataSub,TrainTargetSub,ValidateTargetSub,TestTargetSub)
    printfun3(L_Erms_Test1,Acc_Test1)


# In[64]:

def printfun2(L_Erms_Test,Acc_Test):
    print ('----------Gradient Descent Solution--------------------')
    print ('----------LOGISTIC REGRESSION: Feature Concatenation--------------------')
    #print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))           # Print Erms for training set
    #print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))          # Print Erms for validation set
    #print ("Training Accuracy   = " + str(np.around(max(Acc_Train),5)))           # Print Erms for training set
    #print ("Validation Accuracy= " + str(np.around(max(Acc_Val),5)))          # Print Erms for validation set
    print ("Testing Accuracy= " + str(np.around(max(Acc_Test),5)))   
    print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))         # Print Erms for testing set


# In[65]:

def printfun3(L_Erms_Test1,Acc_Test1):
    print ('----------Gradient Descent Solution--------------------')
    print ('----------LOGISTIC REGRESSION: Feature Subtraction--------------------')
    #print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))           # Print Erms for training set
    #print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))          # Print Erms for validation set
    #print ("Training Accuracy   = " + str(np.around(max(Acc_Train),5)))           # Print Erms for training set
    #print ("Validation Accuracy= " + str(np.around(max(Acc_Val),5)))          # Print Erms for validation set
    print ("Testing Accuracy= " + str(np.around(max(Acc_Test1),5)))   
    print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test1),5)))         # Print Erms for testing set


# In[ ]:


#plt.plot(learningRate, Erms,'ro')
#plt.ylabel('E-RMS Value for Testing')
#plt.xlabel("Learning Rate")
#plt.title("Learning Rate VS E-RMS Plot for Logistic Regression")
#plt.show()


# In[ ]:


#plt.plot(learningRate,acc,'ro')
#plt.ylabel('Testing Accuracy of the model')
#plt.xlabel("Learning Rate")
#plt.title("Learning Rate VS Accuracy Plot for Logistic Regression")
#plt.show()


