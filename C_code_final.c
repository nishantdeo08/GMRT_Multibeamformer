//This C code is connected to the main code, it creates a pipe and send inputs to Python program "Python_code_final", the Python code processes on the data and return the control to this program.

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#define MAX_OUTPUT_SIZE 100

int main() {
    int fd[2]; //File Descriptors
    pid_t pid; // Process ID

    // Definining the inputs which will be sent via pipe
    int num_antennas = 17;
    int freq = 400;
    int RA_h = 13;
    int RA_m = 31;
    float RA_s = 8.29;
    int DEC_deg = 19;
    int DEC_am = 30;
    float DEC_as = 33;
    int Year = 2022;
    int Month = 11;
    int Date = 15;
    int Hour = 5;
    int Min = 18;
    int Sec = 0;
    int MS = 0;

    char name[][5] = {"C00","C01","C02","C03","C04","C05","C06","C08","C09","C10","C11","C12","C13","C14","E02","S01","W01"};

    char output[100];

    // Generates error if there is any issue with pipe formation
    if (pipe(fd) == -1) {
        fprintf(stderr, "Pipe failed");
        return 1;
    }
    
    // Creates the child process (Python Code)
    pid = fork();

    // Raises an error if there is any issue with the subprocess (Child Process) formation
    if (pid < 0) {
        fprintf(stderr,"Fork failed");
        return 1;
    }


    if (pid > 0) {
        close(fd[0]); //Closing the read end of the pipe

        // Writing the inputs from the write end of the pipe
        write(fd[1], &num_antennas, sizeof(num_antennas));
        write(fd[1], &freq, sizeof(freq));
        write(fd[1], &RA_h, sizeof(RA_h));
        write(fd[1], &RA_m, sizeof(RA_m));
        write(fd[1], &RA_s, sizeof(RA_s));
        write(fd[1], &DEC_deg, sizeof(DEC_deg));
        write(fd[1], &DEC_am, sizeof(DEC_am));
        write(fd[1], &DEC_as, sizeof(DEC_as));
        write(fd[1], &Year, sizeof(Year));
        write(fd[1], &Month, sizeof(Month));
        write(fd[1], &Date, sizeof(Date));
        write(fd[1], &Hour, sizeof(Hour));
        write(fd[1], &Min, sizeof(Min));
        write(fd[1], &Sec, sizeof(Sec));
        write(fd[1], &MS, sizeof(MS));
        write(fd[1], name, sizeof(name));
        close(fd[1]); // Closing the write end of the pipe
        
        // Waits for the Child process
	wait(NULL);
    }
    else {
        close(fd[1]); // Closing the write end

	// 2 arrays are defined to store the inputs
        int input_values[15];
        read(fd[0], input_values, sizeof(input_values));
        char name[num_antennas][5];
        read(fd[0], name, sizeof(name));
        close(fd[0]);

	// Pipe is called
        FILE* fp = popen("python3 Python_code_final_optimized.py", "w");
	fprintf(fp, "%d\n", input_values[0]);
        fprintf(fp, "%d\n", input_values[1]);
        fprintf(fp, "%d\n", input_values[2]);
        fprintf(fp, "%d\n", input_values[3]);
        fprintf(fp, "%f\n", *((float*)&input_values[4]));
        fprintf(fp, "%d\n", input_values[5]);
        fprintf(fp, "%d\n", input_values[6]);
        fprintf(fp, "%f\n", *((float*)&input_values[7]));
        fprintf(fp, "%d\n", input_values[8]);
        fprintf(fp, "%d\n", input_values[9]);
        fprintf(fp, "%d\n", input_values[10]);
        fprintf(fp, "%d\n", input_values[11]);
        fprintf(fp, "%d\n", input_values[12]);
        fprintf(fp, "%d\n", input_values[13]);
        fprintf(fp, "%d\n", input_values[14]);
        int i = 0;
        for (i = 0; i < num_antennas; i++) {
            fprintf(fp, "%s\n", name[i]);
        }
        pclose(fp);
    }

    return 0;
}

