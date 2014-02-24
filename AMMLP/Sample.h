#ifndef SAMPLE_H_
#define SAMPLE_H_

#include <vector>

class Sample{
public:
	int getNFeatures(){
		return input.size();
	}

	const std::vector<double>& getInput() const {
		return input;
	}

	void setInput(const std::vector<double>& input) {
		this->input = input;
	}

	std::vector<int> getResult() const {
		return result;
	}

	void setResult(std::vector<int> result) {
		this->result = result;
	}

	double getRResult() const {
		return rResult;
	}

	void setRResult(double rResult) {
		this->rResult = rResult;
	}
	
private:
	std::vector<double> input;
	std::vector<int> result;
	double rResult;
};

#endif /* SAMPLE_H_ */
