"""Lesson outline

If you're familiar with NumPy (esp. the following operations), feel free to skim through this lesson.

    Create a NumPy array:
        from a pandas dataframe: pandas.DataFrame.values
        from a Python sequence: numpy.array
        with constant initial values: numpy.ones, numpy.zeros
        with random values: numpy.random
    Access array attributes: shape, ndim, size, dtype
    Compute statistics: sum, min, max, mean
    Carry out arithmetic operations: add, subtract, multiply, divide
    Measure execution time: time.time, profile
    Manipulate array elements: Using simple indices and slices, integer arrays, boolean arrays

Hit Next to dive in!
"""

nd1 = df1.values # get just the values, no row or column names

nd1[row,col]

nd1[0:3, 1:3] #Rows 1,2,3 and cols 2, and 3

import numpy as NumPy
import time

def test_run():
    print np.array([(2, 3, 4), (5, 6, 7)]) # can take list tuple or sequence

    # Empty Arrays
    print np.empty((5, 4, 3)) # 5 rows 4 cols 3 depth

    # Array full of 1s
    print np.ones((5, 4), dtype=np.int_) # set datatype to uinteger

    print np.zeros()

    # Generate an array full of random floats from 0.0 to 1.0
    print np.random.random((5,4)) # Prefered
    print np.random.rand(5, 4)

    # Sample numbers from a Gaussian (normal) distribution
    print np.random.normal(size=(2,3)) # standard normal (mean=0, std=1)
    print np.random.normal(50, 10, size=(2,3)) # standard normal (mean=50, std=10)

    print np.random.randint(10) # 1 integer from 0 to 10
    print np.random.randint(0, 10) # same as above [low, high)
    print np.random.randint(0, 10, size=5) # 5 numbers in 1d array
    print np.random.randint(0, 10, size=(2,3)) # 2x3 array of random ints

    a = np.random.random((5,4))
    print a.shape # gives the shape (5,4) row col
    print a.shape[0] # num of rows
    print a.shape[1] # num of cols
    print len(a.shape) # Dimension of array (2)
    print a.size # total number of elements 5*4 = 20
    print a.dtype # type of data in array float64

    np.random.seed(693) # seed random number generator to get same random numbers each time
    a = np.random.randint(0, 10, size=(5,4)) # Random numbers from 0 to 10.
    a.sum() # Sum of all elements
    a.sum(axis=0) # sum of each column
    a.sum(axis=1) # sum of each row

    a.min(axis=0) # min of each column
    a.max(axis=1) # max of each row
    a.mean() # average of all elements in column

    # How to time functions
    t1 = time.time()
    print "ML4T"
    t2 = time.time()
    print "The time taken is {} seconds.".format(t2-t1)
    # numpy mean is way faster than manual calculation of mean of multi dimensional arrays

    # Slicing
    print a[:, 0:3:2] # Get all rows and columns 0,1,2 but only in steps of 2 so 0,2
    a[0,:] = 2 # Replace entire row with 2's
    a[:, 3] = [1,2,3,4,5] # Dimension must be the same

    a = np.random.rand(5)
    indices = np.array([1,1,2,3])
    print a[indices]

    # Numpy Masking - Let's get the values less than the mean of an array.
    a = np.random.randint(size=(5,2))
    mean = a.mean()
    # masking
    print a[a<mean]

    # Arithmetic operations
    print a*2
    print a/2
    print a*b # two matrixes of the same shape multiplying the same indices

    # Numpy References https://classroom.udacity.com/courses/ud501/lessons/4134798720/concepts/42767885590923


if __name__ == "__main__":
    test_run()