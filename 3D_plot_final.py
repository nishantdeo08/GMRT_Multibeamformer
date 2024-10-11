import numpy as np
import time
import math
import matplotlib.pyplot as plt
import datetime

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
    
antenna_amplitude = np.ones(23)
antenna_phases = np.zeros(23)
antenna_coordinates = np.zeros((23,3)) 
name = ['C00','C01','C02','C03','C04','C05','C06','C08','C09','C10','C11','C12','C13','C14','E02','E03','E04','S01','S02','S03','W01','W02','W03']
for i in range(23): 
    antenna_coordinates[i] = antenna_loc_names[name[i]]
        

freq = 400*10**6
theta_start = -90
theta_stop = 90
phi_start = 0
phi_stop = 90
theta_num_pts = 511
phi_num_pts = 511

range_deg = 0
range_am = 6
range_as = 0

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
    
RA_h = 13
RA_m = 31
RA_s = 8.29
DEC_deg = 19
DEC_am = 30
DEC_as = 33

   
Y = 2022
M = 11
D = 15
Hr = 5 #10 
Mi = 18 #29 for transit
S = 0
MS = 0

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
dec_array = np.linspace(dec_start,dec_stop,phi_num_pts)
dec_array = dec_array.T

power_pattern = np.zeros((theta_num_pts,phi_num_pts))

# The main "for-loop" which will generate the power pattern!

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

power_pattern = power_pattern_new/max(power_pattern_new)

power_pattern = np.reshape(power_pattern, (theta_num_pts,phi_num_pts))

# Plot contour when 3D radiation pattern is selected
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X,Y = np.meshgrid(dec_array,ra_array)

ax.plot_surface(X,Y,power_pattern, cmap='jet')
ax.set_ylabel('RA')
ax.set_xlabel('DEC')
ax.set_zlabel('Power Pattern')
ax.set_title('3D Plot')

plt.show()