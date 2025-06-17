"""This document gets the TOF measurements from /media/disk_b/standard_software/BNL1TSim/ratdb/BNL1T/PMTINFO.ratdb using distance formula"""

import numpy as np

# goal is to find distances from alpha source to the 30 bottom PMTs and 16 side PMTs and 12 supplemental side PMTs
# note that the supplemental PMTs were added later, nothing else really special
# Also measurements are in mm!
alpha_source_x = -97
alpha_source_y = 13
alpha_source_z = 0

bottom_PMTs_x = [381. ,  381. ,  381. ,  381. ,  190.5,  190.5,  190.5,  190.5, 190.5,  190.5,  190.5,    0. ,    0. ,    0. ,    0. ,    0. , 0. ,    0. ,    0. , -190.5, -190.5, -190.5, -190.5, -190.5, -190.5, -190.5, -381. , -381. , -381. , -381.]
bottom_PMTs_y = [-171.45,  -57.15,   57.15,  171.45, -342.9 , -228.6 , -114.3 , 0.  ,  114.3 ,  228.6 ,  342.9 , -400.05, -285.75, -171.45,-57.15,   57.15,  171.45,  285.75,  400.05, -342.9 , -228.6, -114.3 ,    0.  ,  114.3 ,  228.6 ,  342.9 , -171.45,  -57.15, 57.15,  171.45]
bottom_PMTs_z = [-677.1,-677.1,-677.1,-677.1,-677.1,-677.1,-677.1,-677.1,-677.1,-677.1,-677.1,-677.1,-677.1,-677.1,-677.1,-677.1,-677.1,-677.1,-677.1,-677.1,-677.1,-677.1,-677.1,-677.1,-677.1,-677.1,-677.1,-677.1,-677.1,-677.1]

side_PMTs_x = [-532.955, -532.955, -532.955, -532.955, 532.955, 532.955, 532.955, 532.955, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
side_PMTs_y = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -532.955, -532.955, -532.955, -532.955, 532.955, 532.955, 532.955, 532.955]
side_PMTs_z = [-495.3, -336.55, 222.25, 393.7, -495.3, -336.55, 222.25, 393.7, -495.3, -336.55, 222.25, 393.7, -495.3, -336.55, 222.25, 393.7]

supp_side_PMTs_x = [-376.8561,-376.8561,-376.8561,376.8561,376.8561,376.8561,376.8561,376.8561,376.8561,-376.8561,-376.8561,-376.8561]
supp_side_PMTs_y = [-376.8561,-376.8561,-376.8561,376.8561,376.8561,376.8561,-376.8561,-376.8561,-376.8561,376.8561,376.8561,376.8561]
supp_side_PMTs_z = [-211.0232,-41.1607,128.7018,-211.0232,-41.1607,128.7018,-211.0232,-41.1607,128.7018,-211.0232,-41.1607,128.7018]

# ordered list from google slides from Sunwoo in week 2 notes
bottom_PMT_list = ['adc_b1_ch1', 'adc_b1_ch2', 'adc_b1_ch3', 'adc_b1_ch4', 'adc_b1_ch5', 'adc_b1_ch6', 'adc_b1_ch7', 'adc_b1_ch8', 'adc_b1_ch9', 'adc_b1_ch10', 'adc_b1_ch11', 'adc_b1_ch12', 'adc_b1_ch13', 'adc_b1_ch14', 'adc_b1_ch15', 'adc_b2_ch0', 'adc_b2_ch1', 'adc_b2_ch2', 'adc_b2_ch3', 'adc_b2_ch4', 'adc_b2_ch5', 'adc_b2_ch6', 'adc_b2_ch7', 'adc_b2_ch8', 'adc_b2_ch9', 'adc_b2_ch10', 'adc_b2_ch11', 'adc_b2_ch12', 'adc_b2_ch13', 'adc_b2_ch14']
side_PMT_list = ['adc_b3_ch0', 'adc_b3_ch1', 'adc_b3_ch2', 'adc_b3_ch3', 'adc_b3_ch4', 'adc_b3_ch5', 'adc_b3_ch6', 'adc_b3_ch7', 'adc_b3_ch8', 'adc_b3_ch9', 'adc_b3_ch10', 'adc_b3_ch11', 'adc_b3_ch12', 'adc_b3_ch13', 'adc_b3_ch14', 'adc_b3_ch15']
supp_side_PMT_list = ['adc_b4_ch0', 'adc_b4_ch1', 'adc_b4_ch2', 'adc_b4_ch3', 'adc_b4_ch4', 'adc_b4_ch5', 'adc_b4_ch6', 'adc_b4_ch7', 'adc_b4_ch8', 'adc_b4_ch9', 'adc_b4_ch10', 'adc_b4_ch11']

def distance_formula(x1, y1, z1, x0 = alpha_source_x, y0 = alpha_source_y, z0 = alpha_source_z):
    return np.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)

# let's make a distance dictionary, keeping it in mm
channels_and_distances_mm_dict = {}

for i in range(30):
    bottom_PMT = bottom_PMT_list[i]
    bottom_x = bottom_PMTs_x[i]
    bottom_y = bottom_PMTs_y[i]
    bottom_z = bottom_PMTs_z[i]
    channels_and_distances_mm_dict[bottom_PMT] = distance_formula(bottom_x, bottom_y, bottom_z)

for j in range(16):
    side_PMT = side_PMT_list[j]
    side_x = side_PMTs_x[j]
    side_y = side_PMTs_y[j]
    side_z = side_PMTs_z[j]
    channels_and_distances_mm_dict[side_PMT] = distance_formula(side_x, side_y, side_z)

for k in range(12):
    supp_side_PMT = supp_side_PMT_list[k]
    supp_side_x = supp_side_PMTs_x[k]
    supp_side_y = supp_side_PMTs_y[k]
    supp_side_z = supp_side_PMTs_z[k]
    channels_and_distances_mm_dict[supp_side_PMT] = distance_formula(supp_side_x, supp_side_y, supp_side_z)

print(channels_and_distances_mm_dict)
