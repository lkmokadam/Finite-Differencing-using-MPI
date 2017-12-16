##########################################################################################
# FILE: makefile
# DESCRIPTION:
#
# Command 'make all' will compile p2.c and create an executable file named p2
# Command 'make clean' will remove the executable file named p2 
# LAST REVISED: 8/29/2017
########################################################################################## 	

all: p2_mpi.c
	mpicc -O3 -o p2_mpi p2_mpi.c -lm

clean:
	rm -f *.o p2_mpi
