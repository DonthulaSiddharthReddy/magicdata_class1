import numpy as np 
import pandas as pd #importing libraries

#----------------------------------------------------------Before process---------------------------------------------------------------------------

#-----Loading the data------#

raw=pd.read_csv('/home/siddhu/Miniproject/datasets/magic/dataset.csv')

#------converting into matrix-------#

data=np.array(raw)
Y=data[:,-1]#output set

def convert(Y):#converting g to 1 and h to zero
	y=np.zeros((1,np.size(Y)))
	for i in range(np.size(Y)):
		if Y[i]=='g':
			y[0,i]=1
		else:
			y[0,i]=0
	return y

def deconvert(Y):#converting 1 to g and 0 to h
	y=np.zeros((1,np.size(Y))).astype('str')
	for i in range(np.size(Y)):
		if Y[0,i]==1:
			y[0,i]="g"
		else:
			y[0,i]="h"
	return y


y=convert(Y)#converted matrix 

X=np.delete(data,-1,1)#input set

#----baised featuring-------#

ones=np.array([(1)])
for i in range(1,np.size(Y)):
	ones=np.vstack((ones,np.array([(1)]))) # colomn matrix of ones

X1=np.hstack((ones,X))#baised every feature with one
X1=X1.astype('float64')#making all data of one datatype

#---sigmoid function------#

def sigmoid(z):
	return 1.0/(1.0+(np.exp(-z)))# logistic func

#-----parameters--------#

#theta=np.random.rand(1,np.size(X1[1,0:]))#intializing theta randomly
theta=np.zeros((1,np.size(X1[0,0:])))
alpha=0.0001#learning rate
m=np.size(y)
e=1e-5
ilter=5000

#-------------------------------------------------------------Processing---------------------------------------------------------------------------------

for i in range(ilter):
	#-----linear form-------#

	h=X1@(theta.T) #if datatype of all data is not same the u will get the 'matmul' error

	#-----applying logistic formula------#

	g=sigmoid(h)

	#------gradient decent------------#

	theta=theta-(alpha/m)*(np.sum(X1*(g-(y.T)),axis=0))

	#-----cost function---------#

	J=(np.average((-(y.T)*np.log(g+e))-((1.0-(y.T))*np.log(1-g+e))))

#-------------------------------------------------------------After process-------------------------------------------------------------------------------

print(J)

print(theta)

#------accuracy test--------#

final_theta=theta
prediction=sigmoid(X1@(final_theta.T))
predict=np.array([(0)])
for i in prediction:
	if i<0.5:
		predict=np.hstack((predict,np.array([(0)])))
	else:
		predict=np.hstack((predict,np.array([(1)])))

predict=predict.reshape(1,(m+1))

final_set=predict[:,1:]

f=abs(final_set-y)
accuracy=100-(((np.sum(f))/m)*100)
print(accuracy,"%","accuracy")



#----------------------------------------------------------------Final Touch---------------------------------------------------------------------------------

#------final output-------#

final_output=deconvert(final_set)
print(final_output)

#to see all values of output
#for i in range(m):
#	print(final_output[0,i])
