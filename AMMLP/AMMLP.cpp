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

void AMMLP::train(Sample input) {
	nFeatures = trainingSet[0].getNFeatures();
	init();
	this->initRandomThetas();
	this->initTraining();
	std::cout << "Iniciando el descenso por gradiente..." << std::endl;
	trainByGradient(this->iteraciones, this->alpha);
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
		std::vector<arma::Mat<double>> temp;
		for(int l=0; l<L; l++)
			temp.push_back(this->thetas[l]); // Inicializo la copia
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
			if(vari <= 0.0000001){
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
}

void AMMLP::initTrainingXNOR() {
}

void AMMLP::initRandomThetas() {
}

double AMMLP::sigmoid(double z) {
}

void AMMLP::gradChecking() {
}

double AMMLP::cost() {
}

double AMMLP::subF(arma::mat X) {
}

void AMMLP::pruebaXorBasica() {
}

void AMMLP::init() {
}

void AMMLP::fillTestingY() {
}
