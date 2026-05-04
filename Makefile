SHELL := /bin/bash

MODULES = module purge; module load mpi/2021.12
CC      = mpicc
CFLAGS  = -O2 -std=c11
LIBS    = -llapack -lblas -lm -lgfortran

TARGET  = tsqr_mpi
SRC     = tsqr_mpi.c

all:
	$(MODULES); $(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LIBS)

run:
	$(MODULES); mpirun -np 4 ./$(TARGET)

clean:
	rm -f $(TARGET) *.o tsqr.out tsqr.err