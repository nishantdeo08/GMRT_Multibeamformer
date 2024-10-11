# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 16:39:47 2023

@author: Nishant Deo
"""
import numpy as np
import time
import math
import datetime  
import cv2
import sys
import matplotlib.pyplot as plt
    
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


# Read inputs from the pipe
input_values = []
for _ in range(15):
    input_value = sys.stdin.buffer.readline().strip()
    input_values.append(input_value)

#print("input_values = ",input_values)

antenna_amplitude = np.array([])
antenna_phases = np.array([])
#num_antennas = int(input("Enter the number of antennas: "))
#num_antennas = int(sys.stdin.readline())
num_antennas = int(input_values[0])

# Read input array from the pipe
names = []
for _ in range(15, num_antennas+15):
    name = sys.stdin.buffer.readline().strip()
    name = name.decode(sys.getdefaultencoding()) #To ensure same encoding between C and Python
    names.append(name)

antenna_amplitude = np.ones(num_antennas)
antenna_phases = np.zeros(num_antennas)
antenna_coordinates = np.zeros((num_antennas,3))
best_antenna_coordinates = np.zeros((num_antennas,3))
if num_antennas == 17:
    antennas = ["C00","C01","C02","C03","C04","C05","C06","C08","C09","C10","C11","C12","C13","C14","E02","S01","W01"] 
if num_antennas == 20:
    antennas = ["C00","C01","C02","C03","C04","C05","C06","C08","C09","C10","C11","C12","C13","C14","E02","E03","S01","S02","W01","W02"]

for i in range(num_antennas):
    #name = str(input("Enter the name of the antenna: "))
    antenna_coordinates[i] = antenna_loc_names[names[i]]
    best_antenna_coordinates[i] = antenna_loc_names[antennas[i]]   

#freq = int(input("Enter the frequency in MHz: "))*10**6
freq = int(input_values[1])
freq = int(freq)*10**6
#theta_start = -90
#theta_stop = 90
#phi_start = 0
#phi_stop = 90
theta_num_pts = 511
phi_num_pts = 511

ref_antenna_coordinates = antenna_loc[2, :]

c = 299792458                         # speed of light
lamda = c / freq                      # Lamda
k = 2 * math.pi / lamda

N_temp, width = antenna_coordinates.shape
ant_coord_wrt_ref = np.copy(antenna_coordinates)
best_ant_coord_wrt_ref = np.copy(best_antenna_coordinates)

for i in range(N_temp):
    ant_coord_wrt_ref[i, :] = ant_coord_wrt_ref[i, :] - ref_antenna_coordinates
    best_ant_coord_wrt_ref[i,:] = best_ant_coord_wrt_ref[i,:] - ref_antenna_coordinates

direc_arr = np.copy(ant_coord_wrt_ref)
best_direc_arr = np.copy(best_ant_coord_wrt_ref)
N, width = direc_arr.shape

# Convert array to a string representation
direc_arr_str = np.array2string(direc_arr)

# RA DEC and range declaration  for calibrator field 1830-360

RA_h = int(input_values[2])
RA_m = int(input_values[3])
RA_s = float(input_values[4])
DEC_deg = int(input_values[5])
DEC_am = int(input_values[6])
DEC_as = float(input_values[7])
Y = int(input_values[8])
M = int(input_values[9])
D = int(input_values[10])
Hr = int(input_values[11])
Mi = int(input_values[12])
S = int(input_values[13])
MS = int(input_values[14])

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
    
lat = 19.0919

start = time.time()

# Convert Range deg, arcmin, arcsec to rad
range_rad = ((range_deg+range_am/60+range_as/3600))*(math.pi/180)
#print("Range = ",range_rad)

# Convert IST to LST
t = datetime.datetime(Y, M, D, Hr, Mi, S, MS, tzinfo=datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
p = (t - datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)).total_seconds()
lst = (p - 22809494.62) * 1.0027379094 + 17773
lst = (lst % 86400) / 3600  # Convert LST to hours

# Display the LST in hours, minutes, and seconds
hours, remainder = divmod(round(lst * 3600), 3600)
minutes, seconds = divmod(round(remainder), 60)
lst_hms = [int(hours), int(minutes), int(seconds)]

# lst to rad conversion
lst_hms1 = lst_hms[0]*3600+lst_hms[1]*60+lst_hms[2]
lst_deg = lst_hms1/240
lst_rad = lst_deg * (np.pi/180)
#print("lst_rad  = ",lst_rad)

# Latitude deg to rad
lat_rad = lat*(np.pi/180)

# RA to rad conversion
RA_rad = ((RA_h*3600+RA_m*60+RA_s)/240)*(np.pi/180)
RA_centre = RA_rad
#print("RA = ",RA_centre)

# DEC to rad conversion
DEC_rad = ((DEC_deg + DEC_am/60 +DEC_as/3600)*(np.pi/180))
DEC_centre = DEC_rad
#print("DEC =", DEC_centre)

# Calculate HA
ha = lst_rad - RA_rad
ha_centre = ha
ha_centre_deg = (((ha_centre*180)/np.pi)+720)%360
#print("HA = ",ha_centre)

# Calculate El
el = math.asin(np.sin(lat_rad) * np.sin(DEC_rad) + np.cos(lat_rad) * np.cos(DEC_rad) * np.cos(ha))
el_deg = el*(180/np.pi)
#print("El = ",el)

# Calculate Az
az = math.acos((np.sin(DEC_rad)-np.sin(lat_rad)*np.sin(el)) / (np.cos(lat_rad)*np.cos(el)));
az_deg = az*(180/np.pi)

if ha_centre_deg <= 180:
    az_deg = 360 - ((az * 180) / np.pi)
    az = (az_deg * np.pi) / 180
#print("Az = ",az)

phi_rad = az
theta_deg = 90 - el_deg
theta_rad = 1.5708 - el
#print("Theta_rad = ",theta_rad)

# Antenna-and-Simulation-Parameters
# Enter the phases and amplitudes (feed coefficients) for each antenna.

phase_rms = (antenna_phases * np.pi)/180; # intrinsic phase difference (all 0 if there is no intrinsic phase difference)
phase_rms = phase_rms.T
#amp_i = antenna_amplitude 

#theta_centre = theta_rad
#phi_centre = phi_rad

direc_source = [np.cos(phi_rad)*np.sin(theta_rad),np.sin(phi_rad)*np.sin(theta_rad),np.cos(theta_rad)] # direction vector of the source 
phase_i = np.zeros(N)
best_phase_i = np.zeros(N)

for i in range (N):
    phase_i[i] = -1 * k * np.dot(direc_arr[i, :], direc_source)  # Calculating the respective phases for each antenna, for the given source direction  
    phase_i[i] = phase_i[i] + phase_rms[i]
    best_phase_i[i] = -1 * k * np.dot(best_direc_arr[i, :], direc_source)
   
# Remember:- Take the range of theta and phi around the center of main beam (esp. for GMRT)
# Remember:- "linspace(X1, X2, N) generates N points between X1 and X2"   # Make sure that N is odd!

# RA Range 
ra_start = RA_rad - range_rad/2
ra_stop = RA_rad + range_rad/2
ra_array = np.linspace(ra_start,ra_stop,theta_num_pts)
ra_array = ra_array.T
#print("RA_array = ",ra_array)

# DEC Range
dec_start = DEC_rad - range_rad/2
dec_stop = DEC_rad + range_rad/2
dec_array = np. linspace(dec_start,dec_stop,phi_num_pts)
dec_array = dec_array.T
#print("DEC_array =",dec_array)

power_pattern = np.zeros((theta_num_pts,phi_num_pts))
best_power_pattern = np.zeros((theta_num_pts, phi_num_pts))

# The main "for-loop" which will generate the power pattern!
a1 = time.time()
#Vectorization using numpy array

# Convert ra_array and dec_array to NumPy arrays
ra_array = np.array(ra_array)
dec_array = np.array(dec_array)

# Calculate HA
ha = lst_rad - ra_array

az = []
el = []
az_deg = []

x1 = time.time()

for i in range(len(ha)):
    el = np.append(el, np.arcsin(np.sin(lat_rad) * np.sin(dec_array) + np.cos(lat_rad) * np.cos(dec_array) * np.cos(ha[i])))  # Calculate El

el = np.reshape(el, (theta_num_pts,phi_num_pts))
el = np.transpose(el)

x2 = time.time()

for j in range(len(dec_array)):
    az = np.append(az, np.arccos((np.sin(dec_array[j]) - np.sin(lat_rad) * np.sin(el[j])) / (np.cos(lat_rad) * np.cos(el[j]))))  # Calculate Az

el = el.flatten()

x3 = time.time()

# Check hour angle condition
ha_deg = (((ha * 180) / np.pi) + 720) % 360
ha = ha_deg * np.pi/180 

for i in range(len(ha)):
    if ha[i] <= np.pi:
        #az_deg = np.append(az_deg, 360 - ((az * 180) / np.pi))
        az[i] = 2*np.pi - az[i]

x4 = time.time()
 
phi_rad = az
theta_rad = 1.5708 - el

direc_src = np.array([np.cos(phi_rad) * np.sin(theta_rad), np.sin(phi_rad) * np.sin(theta_rad), np.cos(theta_rad)])
E_arr = 0
best_E_arr = 0

for t in range(N):

    psi = k * np.dot(direc_arr[t, :], direc_src) + phase_i[t]  # Calculating the psi for each antenna, and adding the intrinsic phase difference
    E_arr = E_arr + antenna_amplitude[t] * np.exp(1j * psi)  # Adding the effect of that extra phase to the total E field. i.e adding the signals from antennas with "phase"
    best_psi = k * np.dot(best_direc_arr[t, :], direc_src) + best_phase_i[t]
    best_E_arr = best_E_arr + antenna_amplitude[t] * np.exp(1j * psi)

power_pattern = np.abs(E_arr) ** 2
power_pattern = power_pattern/max(power_pattern)

power_pattern = np.reshape(power_pattern, (theta_num_pts,phi_num_pts))
power_pattern = np.transpose(power_pattern)

best_power_pattern = np.abs(best_E_arr) ** 2
best_power_pattern = best_power_pattern/max(best_power_pattern)

best_power_pattern = np.reshape(best_power_pattern, (theta_num_pts,phi_num_pts))
best_power_pattern = np.transpose(best_power_pattern)

#print("Power_pattern = ",power_pattern)

b1 = time.time()

a2 = time.time()

cntr_level = [0.5*23/num_antennas]

contour_set = plt.contour(dec_array, ra_array, power_pattern, levels=cntr_level)
best_contour_set = plt.contour(dec_array, ra_array, best_power_pattern, levels = cntr_level)

x = []
y = []
best_x = []
best_y = []

#Extracting the contour parameters for gaussian fitting  
for contour_collection in contour_set.collections:
    paths = contour_collection.get_paths()
    for path in paths:
        vertices = path.vertices
        x.extend(vertices[:, 0])
        y.extend(vertices[:, 1])

for contour_collection in best_contour_set.collections:
    paths = contour_collection.get_paths()
    for path in paths:
        vertices = path.vertices
        best_x.extend(vertices[:, 0])
        best_y.extend(vertices[:, 1])

b2 = time.time()

# x_coords and y_coords now contain the coordinates of the entire contour plot)

# Convert X and Y into a single 2D array
a3 = time.time()

points = np.column_stack((x, y))
points = np.float32(points)
best_points = np.column_stack((best_x, best_y))
best_points = np.float32(best_points)

# Fit ellipse using cv2.fitEllipse
ellipse = cv2.fitEllipse(points)
(center, axes, angle) = ellipse
best_ellipse = cv2.fitEllipse(best_points)
(best_center, best_axes, best_angle) = best_ellipse

# Extract ellipse parameters
a = max(best_axes)
b = min(best_axes)
orientation = angle
orientation = 90 + orientation

b3 = time.time()

a4 = time.time()

#t = np.linspace(0, 2*np.pi, 100)

ra_centre = ra_array[int((len(ra_array))/2)]
dec_centre = dec_array[int((len(dec_array))/2)]

theta = np.deg2rad(orientation)

if theta > 180:
    theta = theta - 180

no_beam = 2000
sub_beam = 20
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
    
#2000 Beam centers
for i in range(int(optimum_x/2 + 1)):
    for j in range(int(optimum_y/2) + 1):
        for n in range(-1,2,2):
            for m in range(-1,2,2):
                ellipse_x = dec_centre - n*i*a*np.cos(theta) + m*j*b*np.sin(theta)
                dec_centers = np.append(dec_centers, ellipse_x)
                ellipse_y = ra_centre - n*i*a*np.sin(theta) - m*j*b*np.cos(theta)
                ra_centers = np.append(ra_centers, ellipse_y)

ra_dec_centers = np.column_stack((dec_centers,ra_centers))
ra_dec_centers = np.unique(ra_dec_centers, axis = 0)

ra_centers = ra_dec_centers[:,1]
dec_centers = ra_dec_centers[:,0]

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

b4 = time.time()

a5 = time.time()

#Beam Indexing
s_beam=[]
headings = ["DEC Centers" , "RA Centers" , "Beam Index", "Sub-Beam"]
with open("ra-dec_centers_final.txt", "a") as file:
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
    
b5 = time.time()
end = time.time()

if optimum_x > optimum_y:
    fov = (a*optimum_y*206264.806, b*optimum_x*206264.806)
else:
    fov = (a*optimum_x*206264.806, b*optimum_y*206264.806)

#print("Field of view is: ", fov)

with open("log_file.txt", "a") as file:
    file.write(f"Number of antennas: {num_antennas} \n")
    file.write(f"Antenna Configuration: {names} \n")
    file.write(f"Frequency: {freq} Hz \n")
    file.write(f"RA-DEC: {RA_h} hr {RA_m} min {RA_s} sec {DEC_deg} deg {DEC_am} am {DEC_as} as \n")
    file.write(f"Date & Time: {D}/{M}/{Y} {Hr}:{Mi}:{S}:{MS} \n")
    file.write(f"DEC Array: {dec_array} \n")
    file.write(f"RA Array: {ra_array} \n")
    file.write(f"Power Pattern: {power_pattern} \n")
    file.write(f"Contour Level: {cntr_level} \n")
    file.write(f"Field of View: {fov} \n")

# Source file name
source_file = 'ra-dec_centers_final.txt'

# Destination file name
destination_file = 'log_file.txt'

# Read the content of the source file
with open(source_file, 'r') as source:
    content = source.read()

# Append the content to the destination file
with open(destination_file, 'a') as destination:
    destination.write(content)

print("Power_pattern generation = ",b1 - a1)
print("Contour generation and extraction  = ", b2 - a2)
print("Ellipse Fitting = ", b3 - a3)
print("2000 Beam Generation = ", b4 - a4)
print("File writing = ", b5 - a5)
print("Total_time = ",end-start)
print("El = ", x2 - x1)
print("Az = ", x3 - x2)
print("Ha = ", x4 - x3)

sys.stdout.write("Program executed successfully!")
sys.stdout.flush()
