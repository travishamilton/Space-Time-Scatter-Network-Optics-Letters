# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 22:14:07 2018

@author: travi
"""

userScatter = input("Scatter-points: ")
userTime = input("Time units: ")
userPoints = input("Position units: ")
start = input("Starting Weight Index of Mask: ")	# starting index (1 <--> wN)
end = input("Ending Weight Index of Mask: ")		# ending index (start <--> wN)
timeChanges = input("Number of time changes: ")

address = "forward_model_1D/field_data/"
file_id = f"scatter{userScatter}_T{userTime}_N{userPoints}_start{start}_end{end}_tc{timeChanges}"   #file id
fileName = address + file_id