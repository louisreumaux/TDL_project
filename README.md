**Project Title:** Neural Network vs Neural Tangent Kernel Data Scaling performances

**Description:**
This project comprises a series of Python files aimed at comparing the data scaling performances of a traditional neural network with those of an infinite neural tangent kernel (NTK). The main objective is to analyze the mean squared error (MSE) on the test set as a function of the size of the training set for these two approaches. The main file to execute is `pipeline.py`, which orchestrates the entire process.

**Files:**
1. `pipeline.py`: The main file to execute. It coordinates the training and evaluation process for both approaches.
2. `nn_training.py`: Contains the code for training and evaluating the traditional neural network.
3. `nn_utils.py`: Provides utilities specific to the neural network.
4. `ntk.py`: Implements the computations associated with the neural tangent kernel.
5. `ntk_utils.py`: Utilities for the neural tangent kernel.
6. `utils.py`: Provides general utility functions.

**Execution:**
To run the project, simply execute `pipeline.py`. Make sure to have all necessary dependencies installed.

**Results:**
Upon completion of execution, the program generates a graph comparing the MSE performances for the traditional neural network versus the infinite NTK. Two beta values for the scaling law are also provided.

**Dependencies:**
- Python 3.x
- PyTorch 
- NumPy 
- Matplotlib 
- JAX 
- Neural Tangents
- SciPy

**References:**
- [Limitations of the NTK for Understanding Generalization in Deep Learnin](https://arxiv.org/abs/2206.10012)
