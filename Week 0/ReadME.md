Features

Loads 2D coordinate data from text files
Random initialization of cluster centers
Vectorized distance computation using NumPy broadcasting
Iterative label assignment and center updates
Automatic convergence checking
Visualization of clustered data and centers

Overview of K means code for spice.py :
1. Loads 2-D data points from a file
2. Randomly initializes K cluster centers
3. Repeats :
    E-Step : Assign each point to its nearest cluster
    M-Step : recompute centres as mean of assigned points
4. Stops when labels don't change(convergence)
5. Plots the clustered points and centers
