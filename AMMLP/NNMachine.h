#ifndef NNMACHINE_H_
#define NNMACHINE_H_

#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>

#include "armadillo"
#include "Sample.h"
#include "Utils.h"

typedef arma::mat M2;

class NNMachine {
public:
	NNMachine();
	virtual ~NNMachine();

	void setParameters(char *argv[]);
	void loadTrainingSet(std::string filename);
	void loadTestingSet(std::string filename);
	void loadInput(std::string filename);
	void train();
	void run();
	void test();
	double predict(Sample input);
	void clearTrainingSet();

private:
	int executionMode;
	std::string trainingFile;
	std::string testFile;

	//Aguilañadidos
	std::string inputFile;
	std::string thetasFileName;
	Sample input;
	double threshold;

	int L; //Número de capas
	int nFeatures;
	double lambda;
	double alpha;
	int iteraciones;
	std::vector<int> actualY;
	std::vector<int> predictedY;
	std::vector<int> s_l; //número de nodos por capa
	std::vector<Sample> trainingSet; //Conjunto de samples para el entrenamiento
	std::vector<Sample> testingSet;
	std::vector<arma::mat> thetas; //L-1 matrices
	std::vector<arma::Col<double> > a;
	std::vector<arma::mat> upperDelta;
	std::vector<arma::mat> D;
	std::vector<double> y;

	void forwardPropagate(Sample s);
	void backPropagate();

	void trainByGradient(int iter, double alpha);

	void initTraining();
	void initTrainingXNOR();
	void initRandomThetas();
	double sigmoid(double z);
	void gradChecking();
	double cost();

	//Metodos de pruebas
	void pruebaXorBasica();
	void loadThetas();
	void saveThetas();
	void readThetas(std::vector<std::string> lectura);
	void showThetas();
	void init();

	//Para los tests
	void fillTestingY();

	//AdvancedOptimization
	void trainByOM();
};

#endif /* NNMACHINE_H_ */
