#!/usr/bin/env python3
"""Module doc"""

from pshape import pshape
import torch
import numpy as np

def main():
    """Main function"""
    print(">>> pshape(np.arange(10).reshape(5,2,1), heading=True)")
    print()
    pshape(np.arange(10).reshape(5,2,1), heading=True)
    print()
    print(">>> pshape(np.eye(4), np.arange(10).reshape(5,2,1), heading=True)")
    print()
    pshape(np.eye(4), np.arange(10).reshape(5,2,1), heading=True)
    print()
    print(">>> cool_arr1 = np.random.rand(123,4,2,1)")
    print(">>> cool_arr2 = np.random.rand(123,4,2,2)")
    print(">>> cool_arr3 = np.random.rand(123,4,2,3)")
    cool_arr1 = np.random.rand(123,4,2,1)
    cool_arr2 = np.random.rand(123,4,2,2)
    cool_arr3 = np.random.rand(123,4,2,3)

    print(">>> pshape(cool_arr1, cool_arr2, cool_arr3)")
    print()
    pshape(cool_arr1, cool_arr2, cool_arr3)
    print()
    print(">>> pshape(cool_arr1, np.arange(12).reshape(3,4), cool_arr3)")
    print()
    pshape(cool_arr1, np.arange(12).reshape(3,4), cool_arr3)
    print()

    print(">>> pshape(torch.arange(12).view(3,4), np.arange(12).reshape(3,4), cool_arr3)")
    print()
    pshape(torch.arange(12).view(3,4), np.arange(12).reshape(3,4), cool_arr3, heading=True)
    print()
    

if __name__ == "__main__":
    main()


