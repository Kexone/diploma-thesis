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
	/**
	* @brief Passes the params to the SVM test class.  Overridden method from 3d lib
	* 
	* @return Normalized accuracy
	*/
	double EvaluteCost(std::vector<double> inputs) const override;

	/**
	 * @brief Returns count of params. Overridden method from 3d lib
	 * 
	 * @return m_dim number of dimensions
	 */
	unsigned int NumberOfParameters() const override;

	/**
	* @brief Sets and returns vector of constraints. Overridden method from 3d lib
	*
	* @return constr vector of constraints
	*/
	std::vector<Constraints> GetConstraints() const override;

	unsigned int m_dim;

};


#endif //DIFFERENTIALEVOLUTIONTEST_H
