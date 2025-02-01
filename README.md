# Assignment-2

## Part 1

### Flowchart of the Work

<img src="https://github.com/user-attachments/assets/9dc9c467-8f76-4ad1-9404-39cc7dce1163" width="600" height="400" />

In the above flowchart:
- **Blue** → Training Phase  
- **Red** → Testing Phase  

- Number of codewords in clustering is fixed at **N=500** (as we got the best results with N=500 in Assignment 1).
- Classifier used: **3-layered MLP**

#### Assumptions:
- In a 3-layered MLP, the total number of fully connected (FC) layers is 3.
  - **Input Layer**
  - **First Hidden Layer (fc1)**
  - **Second Hidden Layer (fc2)**
  - **Output Layer (fc3)**
- The input layer is not counted as a trainable layer.

### Results

#### Accuracy Results

<img src="https://github.com/user-attachments/assets/cf5e8d77-be9c-4cb1-8b22-0286ad03a74f" width="600" height="400" />

Given above is **Validation Accuracy vs Hidden Layer Size** for different Activation Functions used in the **3-Layered MLP**.

**Test Accuracy for the Best Model (512_256_Tanh): 61.9048%**

[Colab Notebook - Part 1](https://colab.research.google.com/drive/1lpxFZKFAszEfUatG01pci_Xw7ar2ug39?usp=sharing)

---

## Part 2

### Flowchart of the Work

<img src="https://github.com/user-attachments/assets/50d55e62-731d-4654-8aa2-817019994c11" width="800" height="900" />

In the above flowchart:
- **Blue** → Training Phase  
- **Red** → Testing Phase  

- Classifier used: **3-layered MLP**

#### Assumptions:
- In a 3-layered MLP, the total number of fully connected (FC) layers is 3.
  - **Input Layer**
  - **First Hidden Layer (fc1)**
  - **Second Hidden Layer (fc2)**
  - **Output Layer (fc3)**
- The input layer is not counted as a trainable layer.

### Results

#### Accuracy Results

<img src="https://github.com/user-attachments/assets/cf5e8d77-be9c-4cb1-8b22-0286ad03a74f" width="800" height="900" />

Given above is **Validation Accuracy vs Hidden Layer Size** for different Activation Functions used in the **3-Layered MLP (72 × 72 Input Image)**.

**Test Accuracy for the Best Model (512_256_LeakyRelu): 30.71%**

[Colab Notebook - Part 2](https://colab.research.google.com/drive/12YPBgupNCDemgJSUgy_5T6mAJPDeI4Ui?usp=sharing)
