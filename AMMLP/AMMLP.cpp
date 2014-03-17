/*
 * AMMLP.cpp
 *
 *  Created on: Mar 8, 2014
 *      Author: antonio
 */

#include "AMMLP.h"

AMMLP::AMMLP() {
	// TODO Auto-generated constructor stub

}

AMMLP::~AMMLP() {
	// TODO Auto-generated destructor stub
}

void AMMLP::train(int iteraciones, double alpha) {
	nFeatures = trainingSet[0].getNFeatures();
	init();
	this->initRandomThetas();
	this->initTraining();
	std::cout << "Iniciando el descenso por gradiente..." << std::endl;
	trainByGradient(iteraciones, alpha);
}

double AMMLP::predict(Sample input) {
	forwardPropagate(input);
	std::cerr << this->a[L-1](0) << std::endl;
	return (this->a[L-1](0)>0.5)?1:-1;
}

void AMMLP::forwardPropagate(Sample s) {
	// Implementación vectorizada
	//Seteo el input en la primera capa
	for(int i=0; i<s_l[0]+1; i++){
		if(i==0)
			this->a[0](i) = 1.0; // Seteo el bias
		else this->a[0](i) = s.getInput()[i-1];
	}
	// Forwarding
	for(int l=1; l<L; l++){
//		std::cout << "Propago por la capa: " << l << std::endl;
		// Calculo z para la capa
		arma::mat zL = this->thetas[l-1]*this->a[l-1];
//		std::cout << "Prepara lo capa " << l << " con la z:" << std::endl << zL << std::endl;
		// Y calculo la activación para la capa
		if(l<L-1){
			for(int i=0; i<s_l[l]+1; i++){
				if(i==0)
					this->a[l](i)=1.0;
				else this->a[l](i)=sigmoid(zL(i-1));
			}
		} else {
			for(int i=0; i<s_l[l]; i++)
				this->a[l](i)=sigmoid(zL(i));
		}
//		std::cout << "El valor de la activación es: " << std::endl << a[l] << std::endl;
	}
}

void AMMLP::backPropagate() {
	// Inicializo upperDelta a cero
	this->upperDelta.clear();
	for(int l=0; l<L-1; l++)
		this->upperDelta.push_back(arma::zeros(s_l[l+1], s_l[l]+1));
	for(int s=0; s<trainingSet.size(); s++){
		forwardPropagate(this->trainingSet[s]);
		std::vector<arma::Col<double> > lowerDelta;
		// Inicializo los lowerDelta porque como empiezo al revés al hacer pushback y los índices y los petes...
		for(int l=0; l<L; l++){
			if(l==L-1)
				lowerDelta.push_back(arma::Col<double>(s_l[l]));
			else lowerDelta.push_back(arma::Col<double>(s_l[l]+1));
		}
		for(int l=L-1; l>0; l--){
			if(l==L-1){
				int r = (trainingSet[s].getResult()[0]==-1)?0:1;
//				std::cout << "La activación para la última capa es: " << a[l] << "y el resultado es: " << this->trainingSet[s].getResult()[0] << std::endl;
				lowerDelta[l](0) = a[l](0)-r; // esto es lo que tengo que generalizar para muchas salidas
			} else {
				arma::Col<double> aux;
				arma::mat gP = this->a[l]%(1-a[l]);
//				std::cout << "Para la capa " << l << " tenemos una g': " << std::endl << gP;
				if(l+1 == L-1)
					aux = thetas[l].t()*lowerDelta[l+1];
				else
					aux = thetas[l].t()*lowerDelta[l+1].rows(1,s_l[l+1]);
//				std::cout << "Para la capa " << l << " tenemos un aux': " << std::endl << aux;
				lowerDelta[l] = aux%gP;
			}
//			std::cout << "Para la capa " << l << " he obtenido un lowerDelta de: " << std::endl << lowerDelta[l] << std::endl;
		}
		for(int l=0; l<L-1; l++){
//			std::cout << "Calculo upperDelta para la capa: " << l << " " << std::endl << lowerDelta[l+1] << " " << s_l[l+1] << std::endl;
			if(l+1 == L-1)
				this->upperDelta[l] += lowerDelta[l+1]*this->a[l].t();
			else
				this->upperDelta[l] += lowerDelta[l+1].rows(1,s_l[l+1])*this->a[l].t();
//			std::cout << "El valor de upperDelta es: " << std::endl << upperDelta[l] << "para un theta: " << std::endl << thetas[l] << std::endl;
		}
	}
	// Regularizo y obtengo la D
	for(int l=0; l<L-1; l++){
		for(int i=0; i<s_l[l+1]; i++){
			for(int j=0; j<s_l[l]+1; j++){
				if(j==0)
					this->D[l](i,j) = (this->upperDelta[l](i,j)/this->trainingSet.size());
				else
					this->D[l](i,j) = (this->upperDelta[l](i,j)/this->trainingSet.size())+(lambda*this->thetas[l](i,j));
//				std::cout << "El valor de D para (" << l << "," << i << "," << j << ") es : " << D[l](i,j) << std::endl;
			}
		}
	}
}

void AMMLP::trainByGradient(int iter, double alpha) {
	double pCoste = 0.0;
	for(int it=0; it<iter; it++){
		// Calculo el coste
		backPropagate();
//		std::cout << "Un poco basto pero..." << std::endl;
//		for(int l=0; l<L-1; l++)
//			std::cout << "D para la capa " << l << " vale: " << std::endl << this->D[l];
//		gradChecking();
		double coste = cost();
		std::cout << "Para la iteración " << it << " el coste es: " << coste << std::endl;
		// Recalculo theta para la siguiente iteracion
//		std::vector<arma::Mat<double> > temp;
//		for(int l=0; l<L; l++)
//			temp.push_back(this->thetas[l]); // Inicializo la copia
		for(int l=0; l<L-1; l++){
			/* Aquí me falta la "simultaneous update" */
//			for(int i=0; i<s_l[l+1]; i++)
//				for(int j=0; j<s_l[l]+1; j++)
//					this->thetas[l](i,j)-=alpha*this->D[l](i,j);
			this->thetas[l] = this->thetas[l] - (alpha*this->D[l]);
		}
		double vari = std::abs(pCoste-coste);
		std::cout << "La variación en el coste para la iteración "<< it <<" es de: " << vari << std::endl;
		if(it>0){
			if(vari <= 0.0001){
				std::cout << "Estoy suficientemente entrenado!!!!!!\n";
				break;
			}
//			if(std::isnan(coste))
//				break;
		}
//		std::cout << it << "," << coste << std::endl;
		pCoste = coste;
	}
	std::cout << "Dejo el gradiente con un coste de: " << this->cost() << std::endl;
}

void AMMLP::initTraining() {
	// Init thetas and a
	this->D.clear();
	this->a.clear();
	for(int l=0; l<L; l++){
//		std::cout << "Inicializo a para esta capa " << l << std::endl;
		if(l==L-1){
			this->a.push_back(arma::Col<double>(s_l[l]));
		} else {
			this->a.push_back(arma::Col<double>(s_l[l]+1));
//			std::cout << "Inicializo D para esta capa " << l << std::endl;
			this->D.push_back(arma::mat(s_l[l+1], s_l[l]+1));
		}
	}
}

void AMMLP::initTrainingXNOR() {
	// Init thetas and a
	this->thetas.clear();
	arma::mat thetaL1(2,3);
	thetaL1(0,0)=-30.0;
	thetaL1(0,1)=20.0;
	thetaL1(0,2)=20.0;
	thetaL1(1,0)=10.0;
	thetaL1(1,1)=-20.0;
	thetaL1(1,2)=-10.0;
	this->thetas.push_back(thetaL1);
	arma::mat thetaL2(1,3);
	thetaL2(0,0) = -10.0;
	thetaL2(0,1) = 20.0;
	thetaL2(0,2) = 20.0;
	this->thetas.push_back(thetaL2);
	// Inicializo los a
	for(int l=0; l<L; l++){
		if(l==L-1)
			this->a.push_back(arma::Col<double>(s_l[l]));
		else{
			this->a.push_back(arma::Col<double>(s_l[l]+1));
			this->D.push_back(arma::mat(s_l[l+1], s_l[l]+1));
		}
		this->upperDelta.push_back(arma::mat(s_l[l+1], s_l[l]+1));
	}
}

void AMMLP::initRandomThetas() {
	this->thetas.clear();
	for(int l=0; l<L-1; l++){
		arma::mat thetaL(s_l[l+1], s_l[l]+1);
		for(int i=0; i<s_l[l+1]; i++)
			for(int j=0; j<s_l[l]+1; j++)
				thetaL(i,j) = Utils::uniformRandomDouble(-10.0,10.0);
		this->thetas.push_back(thetaL);
	}
}

double AMMLP::sigmoid(double z) {
	double e = 2.71828182845904523536;
	return 1/(1+pow(e,-z));
}

void AMMLP::gradChecking() {
	std::vector<arma::mat> cThetasPlus;
	std::vector<arma::mat> cThetasMinus;

	double epsilon = 0.001; // Esto tiene que entrar por parámetro
	// Inicializo los thetas random
//	initRandomThetas();
	// Calculo el coste sumando y restando epsilon a nuestro theta
	for(int l=0; l<L-1; l++){
		arma::mat cTPL(s_l[l+1], s_l[l]+1);
		arma::mat cTML(s_l[l+1], s_l[l]+1);
		for(int i=0; i<s_l[l+1]; i++){
			for(int j=0; j<s_l[l]+1; j++){
				this->thetas[l](i,j) += epsilon;
//				std::cout << "Voy por: " << l << "," << i << "," << j << std::endl;
				cTPL(i,j) = cost();
//				std::cout << "Calculo el coste para plus: " << cTPL(i,j) << std::endl;
				this->thetas[l](i,j) -= 2*epsilon;
				cTML(i,j) = cost();
//				std::cout << "Calculo el coste para menos: " << cTML(i,j) << std::endl;
				this->thetas[l](i,j) += epsilon;
			}
		}
		cThetasPlus.push_back(cTPL);
		cThetasMinus.push_back(cTML);
	}
	// Calculo la aproximación
	std::vector<arma::mat> gradAprox;
	for(int l=0; l<L-1; l++){
		gradAprox.push_back(arma::mat(s_l[l+1], s_l[l]+1));
		for(int i=0; i<s_l[l+1]; i++){
			for(int j=0; j<s_l[l]+1; j++){
				gradAprox[l](i,j) = (cThetasPlus[l](i,j)-cThetasMinus[l](i,j))/(2*epsilon);
				std::cout << "Para la capa " << l << " desde el nodo " << i << " al nodo " << j <<
						" tengo un aproximado de: " << gradAprox[l](i,j) << " y una DX " << this->D[l](i,j) << std::endl;
			}
		}
	}
}

double AMMLP::cost() {
	double J = 0.0;
	for(int i=0; i<this->trainingSet.size(); i++){
		int r = (trainingSet[i].getResult()[0]==-1)?0:1;
		this->forwardPropagate(trainingSet[i]);
		J += r*std::log(a[L-1](0)) + (1-r)*std::log(1-a[L-1](0));
	}
	J /= trainingSet.size()*(-1.0);
//	std::cout << "El coste sin regularizar: " << J << std::endl;
	double regularization = 0.0;
	for(int l=0; l<L-1; l++)
		for(int i=0; i<s_l[l]; i++)
			for(int j=0; j<s_l[l+1]; j++)
				regularization+=std::pow(this->thetas[l](j,i),2);
	regularization = (this->lambda/2.0*this->trainingSet.size())*regularization;
	return J+regularization;
}

double AMMLP::subF(arma::mat X) {
}

void AMMLP::pruebaXorBasica() {
	// Unas cosas previas pa tener que probar
	s_l.clear();
	s_l.push_back(2);
	s_l.push_back(2);
	s_l.push_back(1);
	L = s_l.size();
	Sample s;
	std::vector<double> input;
	input.push_back(0);
	input.push_back(1);
	s.setInput(input);
	std::vector<int> result;
	result.push_back(1);
	s.setResult(result);
	this->trainingSet.push_back(s);
	std::cout << "LLego hasta aquí" << std::endl;
	// Con esto hago una thetas y fp and bp
	this->initTrainingXNOR();
	this->backPropagate();
	this->gradChecking();
}

void AMMLP::init() {
	s_l.clear();
	s_l.push_back(this->nFeatures);
	s_l.push_back(this->nFeatures*2);
	s_l.push_back(this->nFeatures);
	s_l.push_back(1);
	L = s_l.size();

	std::cout << "L es " << L << std::endl;
}

void AMMLP::loadTrainingSet(std::string FileName) {
	//	std::cout << "I'm loading training set with the LinRMachine from " << filename << std::endl;
	std::string line;
	std::ifstream trainingFile(FileName.c_str());

	if(trainingFile.is_open()){
		/* Esto procesa un training set en el que X e y están en líneas
		 * alternas. Esperando un X en que cada dimensión esté separada por
		 * puntos y comas */
		while(std::getline(trainingFile,line)) {
			Sample tmp;
			tmp.setInput(Utils::vStovD(Utils::split(line,';')));
			std::getline(trainingFile,line);
			int res = atoi(line.c_str());
			std::vector<int> result; // Esto está hecho para multi-class pero no está implementao después
			result.push_back(res);
			tmp.setResult(result);
			this->trainingSet.push_back(tmp);
		}
		trainingFile.close();
	} else{
		std::cout << "Unable to open file" << std::endl;
	}
}

void AMMLP::fillTestingY() {
}
