# Assignment-2 Part 1

## Flowchart of the Work

![image](https://github.com/user-attachments/assets/9dc9c467-8f76-4ad1-9404-39cc7dce1163)


In the above flowchart:
- **Blue** → Training Phase  
- **Red** → Testing Phase  

Number of codewords in clustering is fixed at N=500 (as we got best results with N=500 in Assignment 1)

Classifier Used above was 3-layered MLP

Note:- Assumption made was in 3 layered MLP the total number of fully connected (FC) layers is 3.

Input Layer
First Hidden Layer (fc1)
Second Hidden Layer (fc2)
Output Layer (fc3)

We did not count input layer as in 3 layers as it is not trainable. It's Just to pass the Inputs.

---

## Results


### Accuracy Results

![image](https://github.com/user-attachments/assets/cf5e8d77-be9c-4cb1-8b22-0286ad03a74f)

Given above is Validation Accuracy vs Hidden Layer Size for different Activation Funcitions used in 3-Layered MLP

**So the Test Accuracy for The Best Model (512_256_LeakyRelu):- 61.9048%** 

# Part 2

## Flowchart of the Work Part 2

<img width="260" alt="image" src="https://github.com/user-attachments/assets/50d55e62-731d-4654-8aa2-817019994c11" />

In the above flowchart:
- **Blue** → Training Phase  
- **Red** → Testing Phase

- Classifier Used above was 3-layered MLP

Note:- Assumption made was in 3 layered MLP the total number of fully connected (FC) layers is 3.

Input Layer
First Hidden Layer (fc1)
Second Hidden Layer (fc2)
Output Layer (fc3)

We did not count input layer as in 3 layers as it is not trainable. It's Just to pass the Inputs.



## Results


### Accuracy Results

![image](https://github.com/user-attachments/assets/cf5e8d77-be9c-4cb1-8b22-0286ad03a74f)

Given above is Validation Accuracy vs Hidden Layer Size for different Activation Funcitions used in 3-Layered MLP (72 x72 Input Image)

**So the Test Accuracy for The Best Model (512_256_LeakyRelu):- 30.71%** 
