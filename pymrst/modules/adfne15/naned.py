"""Port of ADFNE15 ``Naned.m`` drawing-coordinate preparation."""
import numpy as np
def naned(lines):
    lines=np.asarray(lines,float); dimension=lines.shape[1]//2; result=[]
    for axis in range(dimension): result.append(np.column_stack((lines[:,axis],lines[:,axis+dimension],np.full(len(lines),np.nan))).T)
    return tuple(result)
