/*
 * AMMLP.h
 *
 *  Created on: Mar 8, 2014
 *      Author: antonio
 */

#ifndef AMMLP_H_
#define AMMLP_H_

#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "armadillo"
#include "Utils.h"
#include "Sample.h"

class AMMLP {
public:
	AMMLP();
	virtual ~AMMLP();

	void loadTrainingSet(std::string FileName);
	void train(int iteraciones, double alpha);
	double predict(Sample input);

private:
	//Aguilañadidos
	std::string inputFile;
	std::string thetasFileName;
	Sample input;
	double threshold;
	double e;

	int L; //Número de capas
	int nFeatures;
	double lambda;
	double alpha;
	int iteraciones;
	std::vector<Sample> trainingSet;
	std::vector<int> s_l; //número de nodos por capa
	std::vector<arma::mat> thetas; //L-1 matrices
	std::vector<arma::Col<double> > a;
	std::vector<arma::mat> upperDelta;
	std::vector<arma::mat> D;
	std::vector<double> y;

	// Metaplasticity parameters
	double A;
	double B;



	void forwardPropagate(Sample s);
	void backPropagate();

	void trainByGradient(int iter, double alpha);

	void initTraining();
	void initTrainingXNOR();
	void initRandomThetas();
	double sigmoid(double z);
	void gradChecking();
	double cost();

	// Suboptimal functions implementation7
	double subF(Sample s);

	//Metodos de pruebas
	void pruebaXorBasica();
//	void loadThetas();
//	void saveThetas();
//	void readThetas(std::vector<std::string> lectura);
//	void showThetas();
	void init();

	//Para los tests
	void fillTestingY();
};

#endif /* AMMLP_H_ */
