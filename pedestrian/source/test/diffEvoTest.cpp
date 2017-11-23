#include "diffEvoTest.h"

double DiffEvoTest::EvaluteCost(std::vector<double> inputs) const
{
	SvmTest st;
	st.setParams(inputs[0] * 1000, inputs[1], inputs[2], inputs[3]);
	std::stringstream ss;
	ss << "\tNU: " << inputs[1] << std::endl;
	ss << "\tP: " << inputs[2] << std::endl;
	ss << "\tC: " << inputs[3] << std::endl;

	return st.process(ss);
}

unsigned int DiffEvoTest::NumberOfParameters() const
{
	return m_dim;
}

std::vector<de::IOptimizable::Constraints> DiffEvoTest::GetConstraints() const
{
	std::vector<Constraints> constr;

	constr.push_back(Constraints(0.3, 5.0, true));
	constr.push_back(Constraints(0.001, 0.05, true));
	constr.push_back(Constraints(0.001, 0.05, true));
	constr.push_back(Constraints(0.001, 0.05, true));
	//constr.push_back(Constraints(0.3, 1.0, true));
	return constr;
}