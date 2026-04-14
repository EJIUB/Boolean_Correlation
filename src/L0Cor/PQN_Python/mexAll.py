import numpy as np
import ctypes

# Load the shared library
lib = ctypes.CDLL('./project/projectRandom2C.dll')  # Use the appropriate path and extension for your OS

# Define the function signature
lib.projectRandom2C.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.double),  # c
    ctypes.c_double,  # lambda
    ctypes.c_int,  # nVarsTotal
    np.ctypeslib.ndpointer(dtype=np.double)   # p
]

def projectRandom2C(c, lambda_val):
    nVarsTotal = len(c)
    p = np.zeros_like(c)

    # Call the C function
    lib.projectRandom2C(c, lambda_val, nVarsTotal, p)

    return p