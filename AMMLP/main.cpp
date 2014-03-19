/*
 * main.cpp
 *
 *  Created on: Mar 14, 2014
 *      Author: antonio
 */

#include <iostream>
#include "AMMLP.h"

int main(int argc, char** argv){
	std::cout << "Inicializando y procesando data" << std::endl;
	AMMLP ammpl;
	ammpl.loadTrainingSet("Test1.data");
	ammpl.train(1000, 0.001);
	std::cout << "Done" << std::endl;
}

