# GMRT_Multibeamformer
GMRT Multibeam former code

# Description
This is my Summer Project Work at GMRT, Pune. I have developed a Multibeam Former code in Python. The user inputs a file into the main pipeline written in C which contains the required number of antennas, the configuration, RA-DEC, Date and Time and Frequency. Based on this information, the code calculates the beam pattern. The beam pattern is approximated using a ellipse who fitted paramters are noted. Now, 2016 beams are arranged into a rectangular fashion which are further indexed into 24 sub-beams. It is ensured that the field of view of these 2016 beams does not exceed the primary beam's field of view by adjusting the overlap level of the beams accordingly. 

# Code Directory 
1) Real_Time_Multibeam_Solution_plot_final.py: Independent interactive code, the inputs are passed via the terminal. This code plots the beam pattern contour and corresponding fitted ellipse and the 2024 beams and the 24 sub-beams. 
2) Python_code_final_optimized.py: Code integrated with the pipeline, cannot run independently, does not provide any plots, however, the final RA-DEC centers are passed on in a file alongside a log file containing the user inputs and other important parameters.
3) C_code_final.py: This is the C code which integrates the C pipeline with the Python code, ensuring a smooth transition.
4) 3D_plot.py: The power pattern is a 3D quantity, this code provides this 3D plot, used primarily for visualization.

## Installation
To set up the project on your local machine:

1. Clone the repository:
    ```bash
    git clone https://github.com/nishantdeo08/GMRT_Multibeamformer.git
    ```
2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the main script:
    ```bash
    python Real_Time_Multibeam_Solution_plot_final.py
    ```
