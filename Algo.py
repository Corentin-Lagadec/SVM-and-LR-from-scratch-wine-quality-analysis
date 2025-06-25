#!/usr/bin/env python
# coding: utf-8

# ## File Structure and Design Choices
# 
# In this file, you will find the implementation of three main classes: `Algo`, `SVM`, and `LogisticRegression`. 
# 
# - The `Algo` class serves as a base class that contains common functionality and parameters shared by different machine learning algorithms.
# - The `SVM` class inherits from `Algo` and implements the specifics of the Support Vector Machine algorithm.
# - The `LogisticRegression` class also inherits from `Algo` and contains the logic specific to logistic regression.
# 
# ### Justification of Design Choices
# 
# The decision to have `SVM` and `LogisticRegression` inherit from a common base class `Algo` promotes code reuse and modularity. It allows us to centralize shared components such as regularization parameters, kernel functions, and evaluation metrics, which reduces duplication and improves maintainability.
# 
# By grouping all related classes in a single file, we simplify the project structure, making it easier to understand and maintain. This approach also streamlines future extensions since new algorithms can be added by inheriting from `Algo`, ensuring consistency across implementations.
# 
# Overall, this design enhances clarity, encourages clean architecture, and supports scalable development.
# 

# In[3]:


import numpy as np
import metrics


# # Implementation of the base class 'Algo'

# In[4]:


class Algo:
    def __init__(self, lambda_=0.01, max_iter=1000, size_subset = 1, tol=1e-2, kernel=None, rbf_param=2, degree= 2,track=False, step_iter=10, metric=metrics.accuracy):
        # Initialization of common parameters for all algorithms
        self.lambda_ = lambda_ #regularization term
        self.max_iter = max_iter  # maximun of iterations
        self.size_subset = size_subset # size of the  random subset (mini-batch) <= number of examples in the data set S
        self.tol = tol # stop tolerance 
        self.kernel = kernel # kernelized form or not 
        self.rbf_param = rbf_param # "sigma" for kernel = "rbf"
        self.degree =  degree # degree for "poly" kernel
        self.track = track   # if activated draw the evolution of loss and a metric function evolution
        self.step_iter = step_iter # step for the track of loss and a metric function evolution
        self.metric = metric    # metric used for the plot
        
        # Internal attributes initialized later
        self.alpha = None      # value for each support vector
        self.losses = []       # track of the loss
        self.metric_values = [] # track of the metric 
        self.X_train = None
        self.y_train = None
        self.gram_matrix = None
        
    # Linear kernel: dot product
    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2)

    # Polynomial kernel
    def poly_kernel(self, x1, x2):
        return (1 +np.dot(x1,x2)) ** self.degree
    
    # RBF (Gaussian) kernel
    def rbf_kernel(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * self.rbf_param ** 2))
    
    # Compute the kernel between two vectors
    def compute_kernel(self, x1, x2):
        if self.kernel == "linear":
            return self.linear_kernel(x1, x2)
        elif self.kernel == "poly":
            return self.poly_kernel(x1, x2)
        elif self.kernel == "rbf":
            return self.rbf_kernel(x1, x2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    # Compute the full kernel matrix between two datasets
    def compute_kernel_matrix(self, X1, X2):
        if self.kernel == "linear":
            return X1 @ X2.T
        elif self.kernel == "poly":
            return (1 + X1 @ X2.T)**self.degree
        elif self.kernel == "rbf":
            dists = np.sum(X1**2, axis=1)[:, np.newaxis] +                     np.sum(X2**2, axis=1)[np.newaxis, :] -                     2 * (X1 @ X2.T)
            return np.exp(-self.rbf_param * dists)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    # Plot training loss over iterations
    def plot_loss(self):
        list_iter = list(range(self.step_iter, self.step_iter * len(self.losses) + 1, self.step_iter))
        metrics.plot_loss(list_iter, self.losses)
    
    # Plot metric evolution over iterations       
    def plot_metrics(self):
        list_iter = list(range(self.step_iter, self.step_iter * len(self.metric_values) + 1, self.step_iter))
        metrics.plot_metrics(list_iter, self.metric_values, self.metric)
        


# # I) SVM
# 
# ## Mathematical formulation of our Support Vector Machine (SVM)
# 
# We are solving a binary classification problem: we are given training data 
# $(x_1, y_1), \dots, (x_n, y_n) \in \mathbb{R}^d \times \{-1, 1\}$, 
# and we aim to find a classifier that generalizes well.
# 
# ---
# 
# ###  Objective function: Primal SVM (hinge loss + regularization)
# 
# We aim to find a vector $ w \in \mathbb{R}^d $ that minimizes the following objective:
# 
# $$
# \min_{w} \ \frac{\lambda}{2} \|w\|^2 + \frac{1}{n} \sum_{i=1}^n \max(0;\ 1 - y_i (\langle w, x_i \rangle +b))
# $$
# 
# - $ \lambda $: regularization parameter balancing margin size and classification error  
# - $ \max(0,\ 1 - y_i \langle w, x_i \rangle) $: hinge loss function
# - This term $b$ allows the separation hyperplane to be shifted so that it does not have to pass through the origin.
# 
# This is exactly the loss implemented in the `compute_loss` method of the `SVM` class.
# 
# ---
# 
# ### 1) Subgradient of the Objective (for One Sample)
# 
# We use **stochastic gradient descent**: at each iteration, we sample randomly one training example \( (x_i, y_i) \), and compute the subgradient of the objective.
# 
# Let:
# 
# - $ \ell_i(w, b) = \max(0, 1 - y_i(\langle w, x_i \rangle + b)) $
# 
# We consider then two cases:
# 
# ---
# 
# #### Case 1: The hinge loss is active (margin violation)
# 
# If:
# 
# $$
# 1 - y_i(\langle w, x_i \rangle + b) > 0 \quad \Leftrightarrow \quad y_i(\langle w, x_i \rangle + b) < 1
# $$
# 
# Then the subgradient is:
# 
# $$
# \nabla_w J(w) = \lambda w - y_i x_i \\
# \nabla_b J(w) = -y_i
# $$
# 
# ---
# 
# #### Case 2: The hinge loss is inactive
# 
# If:
# 
# $$
# y_i(\langle w, x_i \rangle + b) \geq 1
# $$
# 
# Then the subgradient is:
# 
# $$
# \nabla_w J(w) = \lambda w \\
# \nabla_b J(w) = 0
# $$
# 
# ---
# 
# ### Summary
# 
# This update rule directly follows from the **subgradient of the hinge loss** and the regularization term:
# 
# - The multiplicative term $ (1 - \eta_t \lambda) $ ensures **weight decay**, enforcing margin maximization  
# - The second term $ \eta_t y_t x_t $ corrects the weights if the margin is violated  
# - The bias $ b $ is only updated when a misclassification (or margin violation) occurs
# ###  Pegasos Algorithm (Primal Estimated sub-Gradient Solver for SVM)
# 
# Then we have the following procedure:
# 
# 1. Sample a data point randomly \( (x_t, y_t) \)
# 2. Update the learning rate:  
#    $ \eta_t = \frac{1}{\lambda t} $
# 3. update $ w $ using the subgradient of hinge loss:
#     
#       If $ y_t (\langle w_t, x_t \rangle + b) < 1 $:  
#        $$
#        w_{t+1} = (1 - \eta_t \lambda) w_t + \eta_t y_t x_t
#        $$
#        $$
#        b_{t+1} = b_t + \eta_t y_t
#        $$
#        Else:  
#        $$
#        w_{t+1} = (1 - \eta_t \lambda) w_t
#        $$
#        $$
#        b_{t+1} = b_t
#        $$
# 
# This is the core of the linear part of the `fit()` method.
# 
# ---
# 
# ##  2) Pegasos Update with Mini-Batch Sampling (Batch Size = m)
# 
# Instead of updating the model using a single random example at each iteration, we can generalize Pegasos to use a **mini-batch of \( m \) random examples**. This often leads to faster and more stable convergence.
# 
# ---
# 
# ### Primal Objective Function (Soft Margin SVM)
# 
# We still aim to minimize:
# 
# $$
# \min_{w, b} \quad \frac{\lambda}{2} \|w\|^2 + \frac{1}{n} \sum_{i=1}^n \max(0,\ 1 - y_i(\langle w, x_i \rangle + b))
# $$
# 
# ---
# 
# ### Mini-Batch Pegasos Algorithm
# 
# Let $ A_t \subset \{1, \dots, n\} $ be a random subset (mini-batch) of indices of size \( m \), sampled at iteration \( t \).
# 
# Let:
# 
# $$
# H_t = \{ i \in A_t \ | \ y_i (\langle w_t, x_i \rangle + b_t) < 1 \}
# $$
# 
# These are the examples in the batch that **violate the margin constraint**.
# 
# ---
# 
# ### Gradient Approximation with Batch
# 
# The subgradient estimate becomes:
# 
# $$
# \nabla_w J(w_t) = \lambda w_t - \frac{1}{m} \sum_{i \in H_t} y_i x_i \\
# \nabla_b J(w_t) = -\frac{1}{m} \sum_{i \in H_t} y_i
# $$
# 
# ---
# 
# ### Update Rule
# 
# 
# - 1 Sample a random subset $ A_t \subset \{1, \dots, n\}$
# - 2 Let the learning rate be:
# 
# $$
# \eta_t = \frac{1}{\lambda t}
# $$
# 
# - 3 Update w :
# 
#     - If $ H_t \neq \emptyset $ (some points violate the margin):
# 
#     $$
#     w_{t+1} = (1 - \eta_t \lambda) w_t + \frac{\eta_t}{m} \sum_{i \in H_t} y_i x_i \\
#     b_{t+1} = b_t + \frac{\eta_t}{m} \sum_{i \in H_t} y_i
#     $$
# 
#     - If $ H_t = \emptyset $ (all points satisfy the margin):
# 
#     $$
#     w_{t+1} = (1 - \eta_t \lambda) w_t \\
#     b_{t+1} = b_t
#     $$
# 
# ---
# 
# ### Summary
# 
# Using a mini-batch:
# 
# - Reduces variance in gradient estimates  
# - Allows better use of vectorized operations (more efficient computation)
# - Improves stability and convergence in practice
# 
# ## Tracking the Loss Function in Kernel Pegasos Algorithm
# 
# To monitor the training progress of the Pegasos algorithm, I use the following loss function :
# 
# $$
# \min_{w,b} \quad \frac{\lambda}{2} \|w\|^2 + \frac{1}{n} \sum_{i=1}^n \max \left(0, 1 - y_i (w^\top x_i + b) \right)
# $$
# 
# This consists of two parts:
# 
# 1. **Hinge loss term**:
#    $$
#    \frac{1}{n} \sum_{i=1}^n \max \left(0, 1 - y_i (w^\top x_i + b) \right)
#    $$
#    This term penalizes misclassified points or points within the margin by the hinge function. It measures how well the classifier separates the data with a margin.
# 
# 2. **Regularization term**:
#    $$
#    \frac{\lambda}{2} \|w\|^2 = \frac{\lambda}{2} w^\top w
#    $$
#    This term controls the complexity of the model by penalizing large weights, helping to prevent overfitting.
# 
# Therefore, every 20 training iteration, I compute:
# 
# 1. The prediction $ g(x_i) $ for each training example $ x_i $ via the kernel expansion.
# 2. The average hinge loss over all examples.
# 3. The RKHS norm regularization weighted by $\lambda/2 $.
# 
# This full loss function allows precise tracking of the objective value decrease and assessing convergence while balancing classification performance and model complexity.

# # 3) Kernel Pegasos Algorithm – Mini-Batch Version
# 
# We extend the Pegasos algorithm to use **mini-batches of size ** $ m $  to update the model in a **Reproducing Kernel Hilbert Space (RKHS)**.
# 
# ---
# 
# ## Objective Function
# 
# We aim to minimize the following regularized hinge loss:
# 
# $$
# \min_{f \in \mathcal{H}_K} \quad \frac{\lambda}{2} \| f \|_{\mathcal{H}_K}^2 + \frac{1}{n} \sum_{i=1}^n \max\left(0, 1 - y_i w(x_i)\right)
# $$
# 
# Where in RKHS:
# 
# $$
# f(x) = \sum_{j=1}^{t-1} \alpha_j y_j K(x_j, x)
# $$
# 
# ---
# 
# ## Kernel Pegasos Algorithm (Mini-Batch Size \( m \))
# 
# Let:
# - $ \lambda > 0$: regularization parameter  
# - $ \eta_t = \frac{1}{\lambda t} $: learning rate  
# - $ K(\cdot, \cdot) $: kernel function  
# - $ \mathcal{S} = \{(x_i, y_i)\}_{i=1}^n $: training set
# 
# ---
# 
# ### Mini-Batch Pegasos Algorithm
# 
# Let $ A_t \subset \{1, \dots, n\} $ be a random subset (mini-batch) of indices of size \( m \), sampled at iteration \( t \).
# 
# Let:
# 
# $$
# H_t = \{ i \in A_t \ | \ y_i (\langle w_t, x_i \rangle + b_t) < 1 \}
# $$
# 
# These are the examples in the batch that **violate the margin constraint**.
# 
# ---
# 
# ### Gradient Approximation with Batch
# 
# The subgradient estimate becomes:
# 
# $$
# \nabla_w F(g_t) = \lambda g_t -  \frac{\eta_t}{m} \sum_{i \in H_t} y_i K(x_i,) \cdot \\
# $$
# 
# ---
# 
# ### Steps
# 
# For $ t = 1, 2, \ldots$:
# 
# 1. Sample a random subset $ A_t \subset \{1, \ldots, n\} $ of size $ m $
# 
# 2. Let the learning rate be:
# 
# $$
# \eta_t = \frac{1}{\lambda t}
# $$
# 
# 3. If $ H_t \neq \emptyset $, update:
# 
# $$
# g_{t+1} = \left(1 - \eta_t \lambda \right) g_t + \frac{\eta_t}{m} \sum_{i \in H_t} y_i K(x_i, \cdot)
# $$
# 
# Equivalently, in practice we:
# - Add the $ x_i $ and $ y_i $ of violating points to the support vectors
# - Append the corresponding coefficient $ \alpha_i = \frac{\eta_t}{m} $
# 
# 4. If $ H_t = \emptyset $, perform only the shrinkage:
# 
# $$
# g_{t+1} = \left(1 - \eta_t \lambda \right) g_t
# $$
# 
# ---
# 
# ##  Early Stopping Criterion
# 
# We may stop if the update is small:
# 
# $$
# \| \alpha^{(t)} - \alpha^{(t-1)} \| < \varepsilon
# $$
# 
# Where $ \varepsilon $ is a small threshold.
# 
# ---
# 
# ## Prediction Rule
# 
# To classify a new input $ x $:
# 
# $$
# \hat{y} = \text{sign} \left( \sum_{j} \alpha_j y_j K(x_j, x) \right)
# $$
# 
# ---
# 
# ## Notes
# 
# - Mini-batching improves **stability** and **convergence** over purely stochastic updates.
# - The number of support vectors can grow, but only when margin violations occur.
# 
# We will use three kernels:
# 
# - Linear: $ K(x, x') = x^\top x' $
# - RBF (Gaussian):  
#   $ K(x, x') = \exp\left( -\frac{\|x - x'\|^2}{2\sigma^2} \right) $
# - Polynomial:  
#   $ K(x, x') = (1 + x^\top x')^d $
#   
# ## Tracking the Loss Function in Kernel Pegasos Algorithm
# 
# To monitor the training progress of the Kernel Pegasos algorithm, I use the following complete loss function, which corresponds to the classical SVM objective in a Reproducing Kernel Hilbert Space (RKHS):
# 
# $$
# \mathcal{L}(g) = \frac{1}{n} \sum_{i=1}^n \max\big(0,\, 1 - y_i g(x_i)\big) + \frac{\lambda}{2} \|g\|_{\mathcal{H}_K}^2
# $$
# 
# where:
# - $ g(x) = \sum_{j=1}^n \alpha_j y_j K(x_j, x) $ is the decision function in the RKHS, represented as a weighted sum of kernels between support vectors \( x_j \) and input \( x \).
# - The term $\max(0, 1 - y_i g(x_i))$ is the **hinge loss**, penalizing misclassifications and margin violations.
# - The regularization term $\frac{\lambda}{2} \|g\|_{\mathcal{H}_K}^2$ controls model complexity and is computed as:
# 
# $$
# \|g\|_{\mathcal{H}_K}^2 = \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j K(x_i, x_j)
# $$
# 
# This can be efficiently computed using the precomputed Gram matrix \( K \) of kernel evaluations between training points.
# 
# Therefore, every 20 training iteration, I compute:
# 
# 1. The prediction $ g(x_i) $ for each training example $ x_i $ via the kernel expansion.
# 2. The average hinge loss over all examples.
# 3. The RKHS norm regularization weighted by $\lambda/2 $.
# 
# This full loss function allows precise tracking of the objective value decrease and assessing convergence while balancing classification performance and model complexity.
# 

# In[5]:


class SVM(Algo):

    def __init__(self, lambda_=0.01, max_iter=1000, size_subset = 1, tol=1e-2, kernel=None, rbf_param=2.0, degree=2, track = False, step_iter=10, metric = metrics.accuracy):
        super().__init__(lambda_, max_iter, size_subset, tol, kernel, rbf_param, degree, track, step_iter, metric)
           
    def compute_loss(self, X, y):
        # loss Pegasos only for no kernel version
        hinge_losses = np.maximum(0, 1 - y * (X @ self.w + self.b))
        hinge = np.mean(hinge_losses)
        reg = 0.5 * self.lambda_ * np.dot(self.w, self.w)
        return hinge + reg
    
    def compute_kernel_loss(self, X, y):
        n_samples = X.shape[0]
        f = (self.alpha * y) @ self.gram_matrix
        hinge_losses = np.maximum(0, 1 - y * f)
        hinge = np.mean(hinge_losses)
        reg = 0.5 * self.lambda_ * (self.alpha * y) @ self.gram_matrix @ (self.alpha * y)
        return hinge + reg
    
    def stochastic_gradient_descent(self):
        n_samples, n_features = self.X_train.shape

        for t in range(1, self.max_iter + 1):
            w_old = self.w.copy()

            eta = 1 / (self.lambda_ * t)

            # Random mini-batch of size self.size_subset
            indices = np.random.choice(n_samples, self.size_subset, replace=False)
            X_batch = self.X_train[indices]
            y_batch = self.y_train[indices]

            # Compute set of violators H_t
            margins = y_batch * (X_batch @ self.w + self.b)
            H_t_mask = margins < 1
            H_t_X = X_batch[H_t_mask]
            H_t_y = y_batch[H_t_mask]

            if len(H_t_y) > 0:
                self.w = (1 - eta * self.lambda_) * self.w + (eta / self.size_subset) * np.sum(H_t_y[:, None] * H_t_X, axis=0)
                self.b = self.b + (eta / self.size_subset) * np.sum(H_t_y)
            else:
                self.w = (1 - eta * self.lambda_) * self.w
                # b remains unchanged

            # Tracking loss and metric
            if self.track and t % self.step_iter == 0:
                loss = self.compute_loss(self.X_train, self.y_train)
                self.losses.append(loss)

                if self.metric:
                    predictions = np.sign(self.X_train @ self.w + self.b)
                    score = self.metric(self.y_train, predictions)
                    self.metric_values.append(score)

            delta_w = np.linalg.norm(self.w - w_old)
            norm_w = np.linalg.norm(self.w)
            epsilon = 1e-8
            if t>10 and norm_w>0 and (delta_w / (norm_w + epsilon ) )< self.tol:
                #print(f"Early stopping \n")
                break
        #print(f"stop at iteration {t}")
    
    def stochastic_gradient_descent_kernel(self):
        n_samples = self.X_train.shape[0]
        for t in range(1, self.max_iter + 1):
            alpha_old = self.alpha.copy()
            eta = 1 / (self.lambda_ * t)

            # Select random mini-batch
            A_t = np.random.choice(n_samples, size=self.size_subset, replace=False)

            # Compute prediction for each point in A_t
            K_batch = self.gram_matrix[:, A_t]  # Shape (n_samples, batch_size)
            f_batch = ((self.alpha * self.y_train) @ K_batch).ravel()  # Shape (batch_size,)
            y_batch = self.y_train[A_t].ravel()

            # Identify violating examples
            H_t = A_t[y_batch * f_batch < 1]

            # 1. Apply shrinkage to all alpha values
            self.alpha *= (1 - eta * self.lambda_)

            # 2. Add contribution of violating examples
            self.alpha[H_t] += eta / self.size_subset
                
            # Tracking
            if self.track and t % self.step_iter == 0:
                loss = self.compute_kernel_loss(self.X_train, self.y_train)
                self.losses.append(loss)
                if self.metric is not None:
                    y_pred = self.predict(self.X_train)
                    score = self.metric(self.y_train, y_pred)
                    self.metric_values.append(score)
            
            # Convergence
            if t%20==0:
                delta_alpha = np.linalg.norm(self.alpha - alpha_old)
                norm_alpha = np.linalg.norm(self.alpha)
                epsilon = 1e-8
                if norm_alpha>0 and (delta_alpha / (norm_alpha + epsilon)) < self.tol:
                    #print(f"Early stopping \n")
                    break

        #print(f"stop at iteration {t}")
       
    
    def fit(self, X, y):
        self.X_train = X
        y = y.astype(float).reshape(-1, 1)
        n_samples, n_features = X.shape
        
        # Ensure labels are in {-1, 1}
        unique_labels = np.unique(y)
        if set(unique_labels.flatten()) == {0, 1}:
            y = 2 * y - 1  # transform to {-1, 1}
        
        self.y_train = y.ravel()

        if self.kernel is None:
            # LINEAR PEGASOS WITH MINI-BATCH
            self.w = np.zeros(n_features)
            self.b = 0.0
            self.stochastic_gradient_descent()
           
        else:
            self.alpha = np.zeros(n_samples)  # Alpha vector
            # Precompute kernel matrix
            self.gram_matrix = self.compute_kernel_matrix(X, X)
            self.stochastic_gradient_descent_kernel()

            
    def predict(self, X_test):
        if self.kernel is None:
            preds = np.sign(X_test @ self.w + self.b)
        else:
            support_idx = np.where(self.alpha != 0)[0]
            alpha_sv = self.alpha[support_idx]
            y_sv = self.y_train[support_idx]
            X_sv = self.X_train[support_idx]
            K_test = np.array([[self.compute_kernel(x_sv, x_test)
                                for x_sv in X_sv] for x_test in X_test])  # shape (n_test, n_sv)

            f = K_test @ (alpha_sv * y_sv)  # shape (n_test,)
            preds = np.sign(f)
        return np.where(preds == 0, 1, preds)


# # II) Logistic Regression
# 
# ## Mathematical formulation of our Logistic Regression
# 
# We are solving a binary classification problem: we are given training data 
# $(x_1, y_1), \dots, (x_n, y_n) \in \mathbb{R}^d \times \{0, 1\}$. To align with margin-based methods like SVM, we typically convert the labels to:
# 
# $y_i \in \{−1,+1\}$.
# and we aim to find a classifier that generalizes well.
# 
# ---
# 
# ###  1) Objective function: Primal SVM (logistic loss + regularization)
# 
# We aim to find a vector $ w \in \mathbb{R}^d $ that minimizes the following objective:
# 
# $$
# \min_{w} \ \frac{\lambda}{2} \|w\|^2 + \frac{1}{n} \sum_{i=1}^n \log( 1 + \exp (- y_i (\langle w, x_i \rangle +b)))
# $$
# 
# - $ \lambda $: regularization parameter balancing margin size and classification error  
# - $ \log( 1 + \exp (- y_i (\langle w, x_i \rangle +b))) $: logistic loss function
# - This term $b$ allows the separation hyperplane to be shifted so that it does not have to pass through the origin.
# 
# This is exactly the loss implemented in the `compute_loss` method of the `LogisticRegression` class.
# 
# ---
# 
# ###  Subgradient of the Objective (for One Sample)
# 
# Gradient of the Loss (Per Sample)
# 
# For a single sample $(x_i, y_i)$, define:
# $$
# z_i=y_i(w^⊤x_i+b) 
# $$
# 
# 
# Then, the gradient of the log loss with respect to $w$ and $b$ is:
# 
# We defined $z_i = y_i(w^t x_i + b)$ and then :
# 
# $∇_wJ(w)=λw−y_ix_i⋅σ(−z_i)$
# 
# $∇_bJ(w)=−y_i⋅σ(−z_i)$
# 
# Where $\sigma(z)$ is the sigmoid function:
#     $σ(z)=  \frac{1}{1+exp(−z)}$
#     
# ### Stet for Pegasos Update (One Sample)
# 
# Let $\eta_t = \frac{1}{\lambda t}$ be the learning rate at iteration $t$.
# 
# At each iteration:
# 
# 1. Sample a data point $(x_t, y_t)$
# 
# 2. Compute $z_t = y_t(w^\top x_t + b)$
# 
# 3. Compute gradient:
# $$
# w_{t+1} = (1 - \eta_t \lambda) w_t + \eta_t y_t x_t \cdot \sigma(-z_t) \\
# b_{t+1} = b_t + \eta_t y_t \cdot \sigma(-z_t)
# $$
# 
# ### Step for Pegasos with Mini-Batch (Batch Size m)
# 
# Let $A_t \subset {1, \dots, n}$ be a batch of $m$ indices, randomly selected.
# For each $i \in A_t$, compute $z_i = y_i(w^\top x_i + b)$
# Estimate the gradient:
# $$
# ∇_wJ(w)=λw− \frac{1}{m} \sum_{i=1}^m y_i x_i\sigma(-z_i) \\
# ∇_bJ(w)=\frac{1}{m} \sum_{i \in A_t} y_i\sigma(-z_i)
# $$
# 
#     Update rule:
# $$
# w_{t+1}=(1−\frac{t}{\lambda})w_t+\frac{η_t}{m} \sum_{i \in A_t} y_ix_i\sigma(−z_i) \\
# b_{t+1}=b_t+\frac{η_t}{m} \sum{i \in A_t} yi\sigma(−z_i)
# $$
# 
# ###  Summary
# 
# - The logistic loss gives probabilistic outputs and is smoother than hinge loss.
# 
# - Pegasos updates can be adapted to logistic regression by replacing hinge with log loss gradients.
# 
# - Mini-batches stabilize learning and leverage matrix operations efficiently.

# ## 2) Kernel Logistic Regression — Primal Formulation
# 
# We define the classifier function as a combination of kernels centered at training points:
# 
# $$
# f(x) = \sum_{i=1}^n \alpha_i k(x_i, x)
# $$
# 
# Let $f \in \mathbb{R}^n$ be the vector of predictions on the training set:
# $$
# f = K \alpha
# $$
# where $K \in \mathbb{R}^{n \times n}$ is the Gram matrix with $K_{ij} = k(x_i, x_j)$.
# 
# We can express the regularized logistic loss directly as a function of $\alpha$:
# 
# ---
# 
# ## Justification for Using the Regularization Term $\|\alpha\|^2$ Instead of $\alpha^\top K \alpha$
# 
# In kernelized logistic regression, the regularized objective is often expressed as:
# 
# $$
# J(\alpha) = \frac{\lambda}{2} \alpha^\top K \alpha + \frac{1}{n} \sum_{i=1}^n \log\left(1 + \exp(-y_i f_i)\right)
# $$
# 
# where $K$ is the kernel (Gram) matrix and $f = K \alpha$ represents the predictions on the training set.
# 
# However, it is common in practice to simplify the regularization term to:
# 
# $$
# \frac{\lambda}{2} \|\alpha\|^2 = \frac{\lambda}{2} \alpha^\top \alpha.
# $$
# 
# This simplification is justified for several reasons:
# 
# 1. **Computational Efficiency:**  
#    Computing and manipulating $\alpha^\top K \alpha$ involves the kernel matrix $K$, which is $n \times n$ and can be large. Using $|\alpha\|^2$ reduces computational complexity and memory usage.
# 
# 2. **Bounding the RKHS Norm:**  
#    Since $K$ is positive semi-definite (PSD), it has a largest eigenvalue $\lambda_{\max}(K)$. We have the inequality:
# 
# $$
# \alpha^\top K \alpha \leq \lambda_{\max}(K) \|\alpha\|^2
# $$
# 
#    meaning that controlling $\|\alpha\|^2$ indirectly controls the RKHS norm $\alpha^\top K \alpha$ up to a constant.
# 
# 3. **Practical Regularization:**  
#    Regularizing $\|\alpha\|^2$ corresponds to a ridge penalty on the coefficients $\alpha$, which effectively prevents overfitting by keeping$\alpha$ small. The hyperparameter $\lambda$ tunes the regularization strength.
# 
# 4. **Numerical Stability:**  
#    The matrix $K$ may be ill-conditioned, causing $\alpha^\top K \alpha$ to amplify numerical instabilities. The simpler $\|\alpha\|^2$ regularization is more stable and robust in optimization.
# 
# ---
# 
# **In summary, using $\frac{\lambda}{2} \|\alpha\|^2$ as the regularization term is a practical choice that balances computational efficiency, effective model complexity control, and numerical stability in kernelized logistic regression.**
# 
# 
# ### Primal Objective in Terms of $f$
# $$
# J(\alpha) =  \frac{\lambda}{2} \|\alpha\|^2 + \frac{1}{n} \sum_{i=1}^n \log\big(1 + \exp(-y_i f_i)\big)
# $$
# 
# - The first term is the $\ell_2$ regularization
# - The second term is the empirical logistic loss.
# - We assume here that $K$ is invertible (or pseudo-inverse is used).
# where:
# - $K$ is the kernel matrix: $K_{ij} = K(x_i, x_j)$,
# - $f_i = \sum_{j=1}^n \alpha_j K(x_j, x_i)$.
# 
# Note: in practice, we work with $\alpha$ and compute $f = K \alpha$, so the gradient descent is performed on $\alpha$.
# 
# 
# ---
# 
# ###  Gradient of the Objective
# 
# Let’s define the vector:
# $$
# s_i = \sigma(-y_i f_i) = \frac{1}{1 + \exp(y_i f_i)}
# $$
# 
# Then the gradient of $J(\alpha)$ is:
# 
# $$
# \nabla_\alpha J(\alpha) = \lambda \alpha - \frac{1}{n} \sum_{i=1}^n y_i s_i K_{:,i}
# $$
# 
# Or, in vectorized form:
# 
# $$
# \nabla_\alpha J(\alpha) = \lambda \alpha - \frac{1}{n} K \left( y \odot \sigma(-y \odot (K \alpha)) \right)
# $$
# 
# where:
# - $\odot$ is the element-wise product,
# - $y \in \mathbb{R}^n$ is the vector of labels $\in \{-1, 1\}$,
# - $\sigma(z) = \frac{1}{1 + e^{-z}}$ sigmoid function is applied elementwise.
# 
# ---
# 
# ### Mini-Batch Gradient Descent
# 
# Let $B_t \subset \{1, \dots, n\}$ be a mini-batch of size $m$, sampled at iteration $t$.
# 
# The learning rate is $\eta_t$.
# 
# Steps:
# 1. Compute $f_i = (K \alpha)_i$ for $i \in B_t$
# 2. Compute $s_i = \frac{1}{1 + \exp(y_i f_i)}$
# 3. Gradient estimate:
# 
# $$
# \nabla_\alpha^{(B_t)} = \lambda \alpha - \frac{1}{m} \sum_{i \in B_t} y_i s_i K_{:,i}
# $$
# 
# 4. Update rule:
# 
# $$
# \alpha \leftarrow \alpha - \eta_t \nabla_\alpha^{(B_t)}
# $$
# 
# 
# ###  Prediction Rule
# 
# Once trained, the model predicts a label for a new point $x$ via:
# 
# $$
# f(x) = \sum_{j=1}^n \alpha_j K(x_j, x) + b
# $$
# 
# Then applied the sigmoid function:
# 
# $$
# \sigma(f(x)) = \frac{1}{1 + e^{-f(x)}} \in (0, 1)
# $$
# 
# and then threshold at 0.5:
# 
# $$
# \hat{y} = 
# \begin{cases}
# +1 & \text{if } \sigma(f(x)) \geq 0.5 \\
# -1 & \text{otherwise}
# \end{cases}
# $$
# 

# In[6]:


class LogisticRegression(Algo):
    
    def __init__(self, lambda_=0.01, max_iter=1000, size_subset = 1, tol=1e-2, kernel=None, rbf_param=2.0, degree=2, track = False, step_iter=10, metric = metrics.accuracy):
        super().__init__(lambda_, max_iter, size_subset, tol, kernel, rbf_param, degree, track, step_iter, metric)

    # Sigmoid activation function
    def sigmoid(self,X):   
        import numpy as np  # just in case
        assert callable(np.exp), "np.exp has been overwritten!"
        X = np.array(X)
        X = np.clip(X, -100, 100)
        return 1 / (1 + np.exp(-X))
    
    # Numerically stable version of log(1 + exp(x))
    def log1pexp(self, x):
        x = np.clip(x, -100, 100) 
        return np.where(x > 0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x)))
    
    # Compute regularized logistic loss (linear version)
    def compute_loss(self, X_b):
        """
        Compute regularized logistic loss for labels in {-1, 1}.
        X_b : numpy array, shape (n_samples, n_features + 1), input data with bias term
        returns: scalar, regularized logistic loss
        """
        n_samples = self.X_train.shape[0]
        logits = X_b @ self.w
        yz = self.y_train * logits
        loss_vec = np.log(1 + np.exp(-yz))
        loss_logistic = np.mean(loss_vec)

        reg_term = (self.lambda_ / 2) * np.sum(self.w[1:] ** 2)
        return loss_logistic + reg_term
        
    # Compute regularized logistic loss in kernel space    
    def compute_loss_kernel(self):
        """
        Compute regularized logistic loss in kernel space for labels in {-1, 1}.
        """
        n_samples = self.y_train.shape[0]
        K = self.gram_matrix.copy()
        
        # Compute raw scores f(x_i) = K @ alpha
        f = K @ self.alpha
        yz = self.y_train * f
        loss_vec = self.log1pexp(-yz)
        loss_logistic = np.mean(loss_vec)

        # norm regularization
        regularization = (self.lambda_ / 2) * np.linalg.norm(self.alpha)**2
        
        return loss_logistic + regularization

    # SGD for linear logistic regressio
    def stocashtic_gradient_descent(self,X_b):
        """
        stocashtic_gradient_descent.

        X_b : numpy array, shape (n_samples, n_features + 1), input data with bias term
        """
        n_samples = X_b.shape[0]
        for t in range(1, self.max_iter + 1):
            w_old = self.w.copy()
            eta = 1 / (self.lambda_ * t)
            
            # Mini-batch
            indices = np.random.choice(n_samples, self.size_subset, replace=False)
            X_batch = X_b[indices]
            y_batch = self.y_train[indices]
            
            # Predictions
            z = X_batch @ self.w
            yz = y_batch * z
            sigmoid_term = self.sigmoid(-yz)  # shape (batch_size, 1)
            
            # Gradient of logistic loss
            grad = -(X_batch.T @ (y_batch * sigmoid_term)) / self.size_subset
            
            # Regularisation term (excluding bias b)
            reg = self.lambda_ * self.w
            reg[0] = 0

            self.w -= eta * (grad + reg)
            
            # Tracking loss and metrics
            if self.track and t % self.step_iter == 0:
                loss = self.compute_loss(X_b)
                self.losses.append(loss)
                if self.metric is not None:
                    preds = self.predict(self.X_train)
                    score = self.metric(self.y_train, preds)
                    self.metric_values.append(score)
            
            delta_w = np.linalg.norm(self.w - w_old)
            norm_w = np.linalg.norm(self.w)
            epsilon = 1e-8
            if t>10 and norm_w>0 and delta_w / (norm_w + epsilon ) < self.tol:
                #print(f"Early stopping \n")
                break
        #print(f"stop at iteration {t}")

    # SGD for kernel logistic regression    
    def stochastic_gradient_descent_kernel(self):
        n_samples = self.X_train.shape[0]
        for t in range(1, self.max_iter + 1):
            alpha_old = self.alpha.copy()
            eta = 1 / (self.lambda_ * t)

            indices = np.random.choice(n_samples, self.size_subset, replace=False)
            K_batch = self.gram_matrix[indices, :]  # (batch_size, n_samples)
            y_batch = self.y_train[indices].reshape(-1, 1)  # (batch_size, 1)

            f = K_batch @ self.alpha  # (batch_size, 1)
            yz = y_batch * f  # (batch_size, 1)
            sigmoid_term = self.sigmoid(-yz)  # (batch_size, 1)

            # gradient w.r.t. α: (n_samples, 1)
            grad = -(K_batch.T @ (y_batch * sigmoid_term)) / self.size_subset

            reg = self.lambda_ * (self.alpha)
            self.alpha -= eta * (grad + reg)
            if np.any(np.isnan(self.alpha)) or np.any(np.isinf(self.alpha)):
                print("Numerical instability detected at iteration", t)
                break

            # tracking
            if self.track and t % self.step_iter == 0:
                loss = self.compute_loss_kernel()
                self.losses.append(loss)
                if self.metric is not None:
                    preds = self.predict(self.X_train)
                    score = self.metric(self.y_train, preds)
                    self.metric_values.append(score)

            # convergence      
            if t%20:
                delta_alpha = np.linalg.norm(self.alpha - alpha_old)
                norm_alpha = np.linalg.norm(self.alpha)
                epsilon = 1e-8
                if norm_alpha>0 and (delta_alpha / (norm_alpha + epsilon)) < self.tol:
                    #print("Early stopping\n")
                    break
        #print(f"stop at iteration {t}")
    
    # Fit model to data
    def fit(self, X, y):
        """
        fit function.

        X : numpy array, shape (n_samples, n_features ), input data without bias term b
        y   : numpy array, shape (n_samples, 1), target labels (0 or 1)
        """
        self.X_train = X
        y = y.astype(float).reshape(-1, 1)
        n_samples, n_features = X.shape
        
        # Ensure labels are in {-1, 1}
        unique_labels = np.unique(y)
        if set(unique_labels.flatten()) == {0, 1}:
            y = 2 * y - 1  # transform to {-1, 1}
        
        self.y_train = y
        
        #linear model
        if self.kernel is None:
            # Add bias b to X 
            X_b = np.c_[np.ones((n_samples, 1)), X]
            #y = 

            # Initialization of w (with b included)
            self.w = np.zeros((n_features + 1, 1))
            self.stocashtic_gradient_descent(X_b)
        # Kernel logistic regression
        else:
            # Compute Gram matrix
            self.gram_matrix = self.compute_kernel_matrix(X, X)
            self.alpha = np.zeros((n_samples, 1))  # coefficients alpha

            # launch stochastic gradient on alpha
            self.stochastic_gradient_descent_kernel()
      
    # Predict probabilities for labels 1
    def predict_proba(self, X_test):
        n_samples = X_test.shape[0]
        if self.kernel is None:
            X_b_test = np.c_[np.ones((n_samples, 1)), X_test]
            logits = X_b_test @ self.w
        else:
            K_test = self.compute_kernel_matrix(self.X_train, X_test)
            logits = K_test.T @ self.alpha

        probs = self.sigmoid(logits).ravel()

        return probs
        
    # Predict class labels
    def predict(self, X_test):
        n_samples = X_test.shape[0]
        if self.kernel is None:
            X_b_test = np.c_[np.ones((n_samples, 1)), X_test]
            logits = X_b_test @ self.w
        else:
            K_test = self.compute_kernel_matrix(self.X_train, X_test)
            logits = K_test.T @ self.alpha

        probs = self.sigmoid(logits).ravel()
        
        return np.where(probs >= 0.5, 1, -1)
            
        

