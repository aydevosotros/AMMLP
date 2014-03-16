#ifndef UTILS_H_
#define UTILS_H_

#include <vector>
#include <iostream>
#include <fstream>
#include <time.h>
#include "float.h"
#include "Sample.h"
#include <boost/noncopyable.hpp>
#include <boost/random/random_device.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

//Sin novedad
class Utils{
public:
	static std::vector<double> parseInput(std::string input){
		std::vector<std::string> tmp = split(input,' ');

		std::vector<double> output;

		//Se guarda los 5 valores, sin contar el tiempo unix ni la cabecera
		for(unsigned int i = 2; i < tmp.size(); i++){
			output.push_back(atof(tmp[i].c_str()));
		}

		return output;
	}

	static time_t getTime(int unixtime){
		time_t t = unixtime;

		return t;
	}

	static std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
	    std::stringstream ss(s);
	    std::string item;
	    while (std::getline(ss, item, delim)) {
	        elems.push_back(item);
	    }
	    return elems;
	}

	static std::vector<std::string> split(const std::string &s, char delim) {
	    std::vector<std::string> elems;
	    split(s, delim, elems);
	    return elems;
	}

	static std::vector<double> vStovD(std::vector<std::string> vS){
		std::vector<double> vD;
		for(unsigned int i = 0; i < vS.size(); i++){
			vD.push_back(atof(vS[i].c_str()));
		}
		return vD;
	}

	static double getElement(std::vector<double> vect, std::vector<int> info, int l, int j, int k){
		int index = 0;
		int i;

		for(i = 0; i < l; i++){
			if(i == info.size()-2){
				index += info[i]*info[i+1];
			} else {
				index += info[i]*(info[i+1]-1);
			}
		}

		if(i == info.size()-2){
			index += j*info[i+1];
		} else {
			index += j*(info[i+1]-1);
		}

		index += k;

		return vect[index];
	}

	//falta modificar igual que en el get
	static void setElement(std::vector<double> &vect, std::vector<int> info, int l, int j, int k, double element){
		int index = 0;
		int i;

		for(i = 0; i < l; i++){
			if(i == info.size()-2){
				index += info[i]*info[i+1];
			} else {
				index += info[i]*(info[i+1]-1);
			}
		}

		if(i == info.size()-2){
			index += j*info[i+1];
		} else {
			index += j*(info[i+1]-1);
		}

		index += k;

		vect[index] = element;
	}

	static void plotData(std::vector<double> x, std::vector<double> y){
		std::ofstream myfile;
		myfile.open ("dataset.dat");
		for(int i = 0; i < x.size(); i++)
			myfile << x[i] << " " << y[i] << std::endl;
		myfile.close();
		system("gnuplot -persist -e \"plot './datafile.dat' using 1:2 title 'X' with linespoints, './datafile.dat' using 1:2 title 'Y' with points\"");
	}

	static void scalation(std::vector<Sample> &trainingSet){
		std::vector<double> absoluts = std::vector<double>();

		unsigned int i,j,k;

		for(i = 0; i < trainingSet[0].getInput().size(); i++){
			absoluts.push_back(DBL_MIN);
		}

		for(i = 0; i < trainingSet.size(); i++){
			for(j = 0; j < trainingSet[i].getInput().size(); j++){
				double num = trainingSet[i].getInput()[j];

				if(num < 0){
					num *= -1.0;
				}

				if(num > absoluts[j]){
					absoluts[j] = num;
				}
			}
		}

		std::cout << "Escalo..." << std::endl;

		for(i = 0; i < trainingSet.size(); i++){
			std::cout << "i = " << i << std::endl;

			std::vector<double> aux = trainingSet[i].getInput();

			for(k = 0; k < aux.size(); k++){
				aux[k] = aux[k] / absoluts[k];
			}

			trainingSet[i].setInput(aux);
		}

		std::cout << "Acabo..." << std::endl;
	}

	static double uniformRandomDouble(double min, double max) {
		boost::random::random_device m_generator;
		boost::random::uniform_real_distribution<> dist(min, max);
		return dist(m_generator);
	}
};

#endif /* SAMPLE_H_ */
