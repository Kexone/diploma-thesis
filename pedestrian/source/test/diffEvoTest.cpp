#include "diffEvoTest.h"
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(90, 200);

double DiffEvoTest::EvaluteCost(std::vector<double> inputs) const
{
	SvmTest st;
	st.preprocessing();
	st.setParams((int) inputs[0] * 1000, inputs[1], inputs[2], inputs[3]);

	return (1 - st.process());
}

unsigned int DiffEvoTest::NumberOfParameters() const
{
	return m_dim;
}

std::vector<de::IOptimizable::Constraints> DiffEvoTest::GetConstraints() const
{
	std::vector<Constraints> constr;

	constr.push_back(Constraints(0.1, 3.0, true));
	constr.push_back(Constraints(0.1, 0.5, true));
	constr.push_back(Constraints(0.1, 0.5, true));
	constr.push_back(Constraints(0.1, 0.5, true));
	//constr.push_back(Constraints(0.3, 1.0, true));
	return constr;
}