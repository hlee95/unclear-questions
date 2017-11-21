"""
This file handles the evaluation metrics we report for Part 1 of the project,
which include MAP, MRR, P@1 and P@5.
"""

import numpy as np

class Eval(object):
  def __init__(self, data):
    self.data = data

  def MAP(self):
    sum_p = 0
    for line in self.data:
    	count = 0
    	precision = 0
    	for i in range(len(line)):
    		if line[i] == 1:
    			count += 1
    			precision += float(count)/(i+1)
    	sum_p += precision/count
    return sum_p / len(self.data)

  def MRR(self):
    sum_r = 0
    for line in self.data:
    	for i in range(len(line)):
    		if line[i] == 1:
    			sum_r += 1./(i+1)
    			break
    return sum_r / len(self.data)

  def Precision(self, precision_at):
    arr = self.data[:,:precision_at]
    precisions = np.sum(arr, 1)/float(precision_at)
    return np.sum(precisions)/len(self.data)
