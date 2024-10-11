# -*- coding: utf-8 -*-
"""
Created on Wed May 17 10:55:07 2023

@author: Nishant Deo
"""

import numpy as np
import time
import math
import datetime 
import matplotlib.pyplot as plt
import cv2
    
antenna_loc_names = {'C00':[21.2011264499803, 687.88, -0.0154029849353918],
                       'C01':[42.4644218527599, 326.44, -0.676665878293534],
                       'C02':[0, 0, 0],
                       'C03':[-142.988620202217, -372.71, -4.68855976457728],
                       'C04':[-133.318781651329, -565.97, -7.81328733666289],
                       'C05':[258.908538200707, 67.81, -5.87825201682459],
                       'C06':[231.748518855754, -31.43, -4.82680102805715],
                       'C08':[421.092540542487, 280.68, -7.336293306176],
                       'C09':[159.201542788435, 41.93, -3.68502018746904],
                       'C10':[617.766060686753, -164.85, -11.3244851349197],
                       'C11':[337.405327433799, -603.25, -8.29632564998673],
                       'C12':[669.037699842715, 174.86, -10.0971228164193],
                       'C13':[1177.02497544664, -639.48, -17.234371901297],
                       'C14':[661.883052247108, -473.68, -9.64191101661692],
                       'E02':[-1015.12718769311, 2814.54, -17.1829743141216],
                       'E03':[-2057.6396438768, 4576.04, -36.6862869007426],
                       'E04':[-3083.03001416875, 7780.58, -31.0575825624119],
                       'E05':[-3544.70799352962, 10199.89, -19.9700953236706],
                       'E06':[-4807.53669276363, 12073.31, 0.0950334058773024],
                       'S01':[2960.01769291548, 633.98, -26.6830306996728],
                       'S02':[4519.00301602781, -367.16, -26.6720608790761],
                       'S03':[6767.20131874033, 333.18, -30.5349199956268],
                       'S04':[9490.67161121727, 947.89, -33.2062484281341],
                       'S06':[14148.5953059169, -368.93, -37.0726622572793],
                       'W01':[-624.633339323, -1591.95, 3.19341607038993],
                       'W02':[-1499.08252756531, -3099.44, 8.45122762753789],
                       'W03':[-3063.93593099095, -5200.01, 11.3284810387178],
                       'W04':[-5355.92608380733, -7039.06, 18.972975314919],
                       'W05':[-8272.04682278001, -8103.23, 0.292805790804778],
                       'W06':[-9440.1502488333, -11245.81, -14.1746764559134]}
    

antenna_loc = np.array([[21.2011264499803, 687.88, -0.0154029849353918],
                       [42.4644218527599, 326.44, -0.676665878293534],
                       [0, 0, 0],
                       [-142.988620202217, -372.71, -4.68855976457728],
                       [-133.318781651329, -565.97, -7.81328733666289],
                       [258.908538200707, 67.81, -5.87825201682459],
                       [231.748518855754, -31.43, -4.82680102805715],
                       [421.092540542487, 280.68, -7.336293306176],
                       [159.201542788435, 41.93, -3.68502018746904],
                       [617.766060686753, -164.85, -11.3244851349197],
                       [337.405327433799, -603.25, -8.29632564998673],
                       [669.037699842715, 174.86, -10.0971228164193],
                       [1177.02497544664, -639.48, -17.234371901297],
                       [661.883052247108, -473.68, -9.64191101661692],
                       [-1015.12718769311, 2814.54, -17.1829743141216],
                       [-2057.6396438768, 4576.04, -36.6862869007426],
                       [-3083.03001416875, 7780.58, -31.0575825624119],
                       [-3544.70799352962, 10199.89, -19.9700953236706],
                       [-4807.53669276363, 12073.31, 0.0950334058773024],
                       [2960.01769291548, 633.98, -26.6830306996728],
                       [4519.00301602781, -367.16, -26.6720608790761],
                       [6767.20131874033, 333.18, -30.5349199956268],
                       [9490.67161121727, 947.89, -33.2062484281341],
                       [14148.5953059169, -368.93, -37.0726622572793],
                       [-624.633339323, -1591.95, 3.19341607038993],
                       [-1499.08252756531, -3099.44, 8.45122762753789],
                       [-3063.93593099095, -5200.01, 11.3284810387178],
                       [-5355.92608380733, -7039.06, 18.972975314919],
                       [-8272.04682278001, -8103.23, 0.292805790804778],
                       [-9440.1502488333, -11245.81, -14.1746764559134]])

antenna_amplitude = np.array([])
antenna_phases = np.array([])

print("Do you want to select all the antennas?")
n = int(input("Type 1 for Yes: "))

if n == 1:    
    antenna_amplitude = np.ones(30)
    antenna_phases = np.zeros(30)
    antenna_coordinates = np.zeros((30,3))
    antenna_coordinates= antenna_loc
    
else:
    num_antennas = int(input("Enter the number of antennas: "))
    antenna_amplitude = np.ones(num_antennas)
    antenna_phases = np.zeros(num_antennas)
    antenna_coordinates = np.zeros((num_antennas,3)) 
    for i in range(num_antennas):
        name = str(input("Enter the name of the antenna: "))
        antenna_coordinates[i] = antenna_loc_names[name]
        

freq = int(input("Enter the frequency in MHz: "))*10**6
theta_start = -90
theta_stop = 90
phi_start = 0
phi_stop = 90
theta_num_pts = 511
phi_num_pts = 511

ref_antenna_coordinates = antenna_loc[2, :]

c = 299792458                         # speed of light
lamda = c / freq                      # Lamda
k = 2 * math.pi / lamda

N_temp, width = antenna_coordinates.shape
ant_coord_wrt_ref = np.copy(antenna_coordinates)

for i in range(N_temp):
    ant_coord_wrt_ref[i, :] = ant_coord_wrt_ref[i, :] - ref_antenna_coordinates

direc_arr = np.copy(ant_coord_wrt_ref)
N, width = direc_arr.shape

# Convert array to a string representation
direc_arr_str = np.array2string(direc_arr)

# Open the file in write mode
with open("ant_coord_wrt_ref.txt", "w") as f:
    # Write the array data to the file
    f.write(direc_arr_str)
    
# RA DEC and range declaration  for calibrator field 1830-360
    
print("Do you want the standard RA-DEC configuration?")
std = int(input("Type 1 for Yes: "))

if std == 1:
    RA_h = 18
    RA_m = 32
    RA_s = 26.81
    DEC_deg = -36
    DEC_am = 1
    DEC_as = 29.92
    
else:
    
    RA_h = float(input("Enter RA hours: "))  # 0-24 hrs
    RA_m = float(input("Enter RA min: "))    # mins
    RA_s = float(input("Enter RA sec: "))    # secs
    
    DEC_deg = float(input("Enter DEC degrees: ")) # deg
    DEC_am = float(input("Enter DEC arcmin: "))   # arc min
    DEC_as = float(input("Enter DEC arcsec: "))   # arc sec

print("Do you want to manually enter the date and time?")
dt = int(input("Type 1 for Yes: "))

if dt == 1:
    Y = int(input("Enter the year: "))
    M = int(input("Enter the month: "))
    D = int(input("Enter the date: "))
    Hr = int(input("Enter the hour: "))
    Mi = int(input("Enter the minute: "))
    S = int(input("Enter the second: "))
    MS = int(input("Enter the microsecond: "))
    
else:

    current_time = datetime.datetime.today()
    
    Y = current_time.year
    M = current_time.month
    D = current_time.day
    Hr = current_time.hour
    Mi = current_time.minute
    S = current_time.second
    MS = current_time.microsecond
    
start = time.time()

# Range of RA-DEC
if freq > 300*10**6 or freq <= 500*10**6:
    range_deg = 0
    range_am = 6
    range_as = 0

elif freq > 500*10**6 or freq <= 900*10**6:
    range_deg = 0
    range_am = 4
    range_as = 0
    
elif freq > 900*10**6 or freq <= 1500*10**6:
    range_deg = 0
    range_am = 2
    range_as = 0

else:
    print("Enter the range in dec,am,as format: ")
    range_dec = int(input("range_dec = "))
    range_am = int(input("range_am = "))
    range_as = int(input("range_as = "))
    
lat = 19.0919

# Convert Range deg, arcmin, arcsec to rad
range_rad = ((range_deg+range_am/60+range_as/3600))*(math.pi/180)

# Convert IST to LST
t = datetime.datetime(Y, M, D, Hr, Mi, S, MS, tzinfo=datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
p = (t - datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)).total_seconds()
lst = (p - 22809494.62) * 1.0027379094 + 17773
lst = (lst % 86400) / 3600  # Convert LST to hours

# Display the LST in hours, minutes, and seconds
hours, remainder = divmod(lst * 3600, 3600)
minutes, seconds = divmod(remainder, 60)
lst_hms = [int(hours), int(minutes), int(seconds)]

# lst to rad conversion
lst_hms1 = lst_hms[0]*3600+lst_hms[1]*60+lst_hms[2]
lst_deg = lst_hms1/240
lst_rad = lst_deg * (np.pi/180)

# Latitude deg to rad
lat_rad = lat*(np.pi/180)

# RA to rad conversion
RA_rad = ((RA_h*3600+RA_m*60+RA_s)/240)*(np.pi/180)
RA_centre = RA_rad

# DEC to rad conversion
DEC_rad = ((DEC_deg + DEC_am/60 +DEC_as/3600)*(np.pi/180))
DEC_centre = DEC_rad

# Calculate HA
ha = lst_rad - RA_rad
ha_centre = ha
ha_centre_deg = (((ha_centre*180)/np.pi)+720)%360

# Calculate El
el = math.asin(np.sin(lat_rad) * np.sin(DEC_rad) + np.cos(lat_rad) * np.cos(DEC_rad) * np.cos(ha))
el_deg = el*(180/np.pi)

# Calculate Az
az = math.acos((np.sin(DEC_rad)-np.sin(lat_rad)*np.sin(el)) / (np.cos(lat_rad)*np.cos(el)));
az_deg = az*(180/np.pi)

if ha_centre_deg <= 180:
    az_deg = 360 - ((az * 180) / np.pi)
    az = (az_deg * np.pi) / 180

phi_rad = az
theta_deg = 90 - el
theta_rad = 1.5708 - el

# Antenna-and-Simulation-Parameters
# Enter the phases and amplitudes (feed coefficients) for each antenna.

phase_rms = (antenna_phases * np.pi)/180; # intrinsic phase difference (all 0 if there is no intrinsic phase difference)
phase_rms = phase_rms.T
#amp_i = antenna_amplitude

#theta_centre = theta_rad
#phi_centre = phi_rad

direc_source = [np.cos(phi_rad)*np.sin(theta_rad),np.sin(phi_rad)*np.sin(theta_rad),np.cos(theta_rad)] # direction vector of the source 
phase_i = np.zeros(N)

for i in range (N):
    phase_i[i] = -1 * k * np.dot(direc_arr[i, :], direc_source)  # Calculating the respective phases for each antenna, for the given source direction  
    phase_i[i] = phase_i[i] + phase_rms[i]
   
# Remember:- Take the range of theta and phi around the center of main beam (esp. for GMRT)
# Remember:- "linspace(X1, X2, N) generates N points between X1 and X2"   # Make sure that N is odd!

# RA Range
# RA_rad = RA_rad 
ra_start = RA_rad - range_rad/2
ra_stop = RA_rad + range_rad/2
ra_array = np.linspace(ra_start,ra_stop,theta_num_pts)
ra_array = ra_array.T


# Remember:- Take the range of theta and phi around the center of main beam (esp. for GMRT)
# Remember:- "linspace(X1, X2, N) generates N points between X1 and X2"    # Make sure that N is odd!

# DEC Range
# DEC_rad = DEC_rad 
dec_start = DEC_rad - range_rad/2
dec_stop = DEC_rad + range_rad/2
dec_array = np. linspace(dec_start,dec_stop,phi_num_pts)
dec_array = dec_array.T

power_pattern = np.zeros((theta_num_pts,phi_num_pts))

# The main "for-loop" which will generate the power pattern!

a1 = time.time()

for a in range(theta_num_pts):

    for b in range(phi_num_pts):
        # Calculate HA
        ha = lst_rad - ra_array[a]

        # RA to rad conversion
        RA_rad = ra_array[a]

        # DEC to rad conversion
        DEC_rad = dec_array[b]

        # Calculate El
        el = math.asin(np.sin(lat_rad) * np.sin(DEC_rad) + np.cos(lat_rad) * np.cos(DEC_rad) * np.cos(ha))

        # Calculate Az
        az = math.acos((np.sin(DEC_rad) - np.sin(lat_rad) * np.sin(el)) / (np.cos(lat_rad) * np.cos(el)))

        # Check hour angle condition
        ha_deg = (((ha * 180) / np.pi) + 720) % 360
        if ha_deg <= 180:

            az_deg = 360 - ((az * 180) / np.pi)
            az = (az_deg * np.pi) / 180

        phi_rad = az
        theta_rad = 1.5708 - el

        direc_src = [np.cos(phi_rad) * np.sin(theta_rad), np.sin(phi_rad) * np.sin(theta_rad), np.cos(theta_rad)]  # Direction of source, calculated at each (theta,phi) combination
        E_arr = 0

        for t in range(N):

            psi = k * np.dot(direc_arr[t, :], direc_src) + phase_i[t]  # Calculating the psi for each antenna, and adding the intrinsic phase difference
            E_arr = E_arr + antenna_amplitude[i] * np.exp(1j * psi)  # Adding the effect of that extra phase to the total E field. i.e adding the signals from antennas with "phase"

        power_pattern[a, b] = np.abs(E_arr) ** 2  # square of E field, i.e Power pattern
            
power_pattern_new = power_pattern.flatten()

power_pattern = power_pattern/max(power_pattern_new)

b1 = time.time()

a2 = time.time()

cntr_level = [0.5]

contour_set = plt.contour(dec_array, ra_array, power_pattern, levels=cntr_level, linestyles='dashed', colors='blue')

x = []
y = []

for contour_collection in contour_set.collections:
    paths = contour_collection.get_paths()
    for path in paths:
        vertices = path.vertices
        x.extend(vertices[:, 0])
        y.extend(vertices[:, 1])

# x_coords and y_coords now contain the coordinates of the entire contour plot

plt.xlabel('DEC')
plt.ylabel('RA')
plt.grid(True)

# Assuming x and y are the arrays of x and y coordinates of the data points
points = np.column_stack((x, y))
points = np.float32(points)

# Fit ellipse using cv2.fitEllipse
ellipse = cv2.fitEllipse(points)
(center, axes, angle) = ellipse

# Extract ellipse parameters
a = max(axes)
b = min(axes)
orientation = angle

orientation = 90 + orientation
theta = np.deg2rad(orientation)
# Create an array of angles from 0 to 2*pi
angles = np.linspace(0, 2 * np.pi, 100)

# Compute the x and y coordinates of the ellipse boundary
ellipse_x = center[0] + a / 2 * np.cos(angles) * np.cos(np.deg2rad(orientation)) - b / 2 * np.sin(angles) * np.sin(np.deg2rad(orientation))
ellipse_y = center[1] + a / 2 * np.cos(angles) * np.sin(np.deg2rad(orientation)) + b / 2 * np.sin(angles) * np.cos(np.deg2rad(orientation))

# Plot the ellipse
plt.plot(ellipse_x, ellipse_y, color='blue', label='Ellipse')
plt.scatter(x, y, color='red', label='Data Points')
plt.axis('equal')
plt.legend()
plt.show()

print("The major axis is: ", a)
print("The minor axis is: ", b)
print("The orientation angle is:",orientation)

b3 = time.time()

a4 = time.time()

ra_centre = ra_array[int((len(ra_array))/2)]
dec_centre = dec_array[int((len(dec_array))/2)]

if theta > 180:
    theta = theta - 180

no_beam = int(input("Enter the number of beams: "))
sub_beam = int(input("Enter the number of sub-beams: "))

if theta > 45 and theta < 135:
    optimum_y = int(math.sqrt(no_beam*b/a))
    optimum_x = int(math.sqrt(no_beam*a/b))

else:
    optimum_x = int(math.sqrt(no_beam*b/a))
    optimum_y = int(math.sqrt(no_beam*a/b))
    
    
ra_centers = np.array([])
dec_centers = np.array([])

    
#Since, in the plotting step, we want equal number of ellipses on each side
if optimum_x % 2 == 1:
    optimum_x = optimum_x + 1
    
if optimum_y % 2 == 1:
    optimum_y = optimum_y + 1
    
while (optimum_x + 1)* (optimum_y + 1) < no_beam:
    if optimum_x > optimum_y:
        optimum_x = optimum_x + 2
    else:
        optimum_y = optimum_y + 2

#print("optimum_x = ",optimum_x)
#print("optimum_y = ",optimum_y)
    
#2000 Ellipse Plotting
for i in range(int(optimum_x/2)+1):
    for j in range(int(optimum_y/2)+1):
        for n in range(-1,2,2):
            for m in range(-1,2,2):
                ellipse_x = dec_centre - n*i*a*np.cos(theta) + m*j*b*np.sin(theta) + (a/2) * np.cos(angles) * np.cos(np.deg2rad(orientation)) - (b/2) * np.sin(angles) * np.sin(np.deg2rad(orientation))
                ellipse_y = ra_centre - n*i*a*np.sin(theta) - m*j*b*np.cos(theta) + (a/2) * np.cos(angles) * np.sin(np.deg2rad(orientation)) + (b/2) * np.sin(angles) * np.cos(np.deg2rad(orientation))
                plt.plot(ellipse_x,ellipse_y)

plt.xlabel("DEC")
plt.ylabel("RA")
plt.title("2000 Beams formation")
plt.axis('equal')
plt.grid(True)

#2000 Beam centers
for i in range(int(optimum_x/2 + 1)):
    for j in range(int(optimum_y/2) + 1):
        for n in range(-1,2,2):
            for m in range(-1,2,2):
                ellipse_center_x = dec_centre - n*i*a*np.cos(theta) + m*j*b*np.sin(theta)
                dec_centers = np.append(dec_centers, ellipse_center_x)
                ellipse_center_y = ra_centre - n*i*a*np.sin(theta) - m*j*b*np.cos(theta)
                ra_centers = np.append(ra_centers, ellipse_center_y)

ra_dec_centers = np.column_stack((dec_centers,ra_centers))
ra_dec_centers = np.unique(ra_dec_centers, axis = 0)

print("RA-DEC_centers = ", ra_dec_centers)

ra_centers = ra_dec_centers[:,1]
dec_centers = ra_dec_centers[:,0]

plt.scatter(dec_centers, ra_centers)
plt.show()

#print("RA_centre = ",ra_centers)          
#print("DEC_centre = ",dec_centers)
dec_array_degamas = []
ra_array_hms = []
                
#Conversion into required format
ra_centers_conv = ra_centers * (180 / np.pi) * 240

ra_centers_hr1 = (ra_centers_conv/3600)
ra_centers_hr = ra_centers_hr1.astype(int)
ra_centers_min1 = ((ra_centers_conv % 3600)/60)
ra_centers_min = ra_centers_min1.astype(int)
ra_centers_sec = (ra_centers_conv % 3600) % 60

dec_centers_conv = dec_centers * (180 / np.pi)

dec_centers_deg = dec_centers_conv.astype(int)
dec_centers_am1 = (dec_centers_conv - dec_centers_deg)*60
dec_centers_am = dec_centers_am1.astype(int)
dec_centers_as = (dec_centers_am1 - dec_centers_am)*60
                
actual_beam = (optimum_x+1)*(optimum_y+1)
#print("Actual No. Beam = ",actual_beam)
#for i in range((optimum_x+1)*(optimum_y+1)):
for i in range(no_beam):
    ra_array_hms.append(f"{ra_centers_hr[i]}hr{ra_centers_min[i]}min{ra_centers_sec[i]}sec")
    dec_array_degamas.append(f"{dec_centers_deg[i]}deg{dec_centers_am[i]}am{dec_centers_as[i]}as")
#print("Ra_array_hms = ", ra_array_hms)
#print("Dec_array_degamas = ",dec_array_degamas)    
'''                
# Vectorization
i_range = np.arange(optimum_x)
j_range = np.arange(optimum_y)
n_range = np.array([-1, 1])
m_range = np.array([-1, 1])

i_grid, j_grid, n_grid, m_grid = np.meshgrid(i_range, j_range, n_range, m_range, indexing='ij')

ellipse_x = ra_centre - n_grid * i_grid * a * np.cos(theta) + m_grid * j_grid * b * np.sin(theta)
ra_centers = ellipse_x.reshape((optimum_x * optimum_y))

ellipse_y = dec_centre - n_grid * i_grid * a * np.sin(theta) - m_grid * j_grid * b * np.cos(theta)
dec_centers = ellipse_y.reshape((optimum_x * optimum_y))'''

b4 = time.time()

a5 = time.time()

'''
# Save arrays to a file
data = np.column_stack((dec_array_degamas, ra_array_hms))
headings = ["DEC centers","RA_Centers","Beam Index"]
np.savetxt('ra-dec_centers.txt',data, delimiter='\t', fmt='%s', header='\t\t\t'.join(headings), comments='')
#np.savetxt('output.txt', np.column_stack((ra_array_hms, dec_array_degamas)), fmt='%s', delimiter='\t')
'''
#Beam Indexing
s_beam=[]
headings = ["DEC Centers" , "RA Centers" , "Beam Index", "Sub-Beam"]
with open("ra-dec_centers.txt", "a") as file:
   np.savetxt(file, [headings], delimiter = "\t\t\t", fmt = "%s")
   
   for i in range(sub_beam):
      dec_array_degamas_split = np.array_split(dec_array_degamas,sub_beam)[i]
      ra_array_hms_split = np.array_split(ra_array_hms,sub_beam)[i]
      #print("RA_array_split = ",len(ra_array_hms_split))
      #print("DEC_array_split = ",len(dec_array_degamas_split))
      index = np.arange(no_beam/sub_beam)
      #print("Index = ",len(index))
      s_beam = [i]*len(index)
      #print("S_Beam = ",len(s_beam))
      data = np.column_stack((dec_array_degamas_split,ra_array_hms_split,index,s_beam))
      np.savetxt(file,data,delimiter = "\t", fmt = "%s")


plt.show()

b5 = time.time()
end = time.time()

#print("Power_pattern generation = ",b1 - a1)
#print("Contour generation and extraction  = ", b2 - a2)
#print("Ellipse Fitting = ", b3 - a3)
#print("2000 Beam Generation = ", b4 - a4)
#print("File writing = ", b5 - a5)
#print("Total_time = ",end-start)
