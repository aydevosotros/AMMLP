GCC=g++
CFLAGS=
INC_DIR=include/
LIB_DIR=lib/ 
MAINFILE=main
OBJFILE=AMMLP
LIBS=-lboost_random -larmadillo -lCGAL
MORE_INCLUDES= NNMachine.cpp AMMLP.cpp

all: $(MAINFILE) 

$(MAINFILE): $(MAINFILE).cpp
	$(GCC) $(CFLAGS) $(MAINFILE).cpp -o $(OBJFILE) -I $(INC_DIR) $(MORE_INCLUDES) -L$(LIB_DIR) $(LIBS) -frounding-math -fopenmp
