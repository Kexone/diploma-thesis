#ifndef DIFFERENTIALEVOLUTIONTEST_H
#define DIFFERENTIALEVOLUTIONTEST_H
#include <vector>
#include "../../3dparty/de/DifferentialEvolution.h"
#include "svmTest.h"
class DiffEvoTest : public de::IOptimizable
{
public:
	explicit DiffEvoTest(unsigned dim)
		: m_dim(dim)
	{
	}

private:
	double EvaluteCost(std::vector<double> inputs) const override;
	unsigned int NumberOfParameters() const override;
	std::vector<Constraints> GetConstraints() const override;

	unsigned int m_dim;

};


#endif //DIFFERENTIALEVOLUTIONTEST_H
