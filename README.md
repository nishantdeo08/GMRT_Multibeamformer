# GMRT_Multibeamformer
GMRT Multibeam former code

# Description
This is my Summer Project Work at GMRT, Pune. I have developed a Multibeam Former code in Python. The user inputs a file into the main pipeline written in C which contains the required number of antennas, the configuration, RA-DEC, Date and Time and Frequency. Based on this information, the code calculates the beam pattern. The beam pattern is approximated using a ellipse who fitted paramters are noted. Now, 2016 beams are arranged into a rectangular fashion which are further indexed into 24 sub-beams. It is ensured that the field of view of these 2016 beams does not exceed the primary beam's field of view by adjusting the overlap level of the beams accordingly. 

# Code Directory 
