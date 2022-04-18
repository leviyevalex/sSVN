// 3rd time's the charm, goddamn

#include <iostream>
#include <cmath>

using namespace std;

void wolf_speak() {
	cout << "woof woof" << endl;
}

int sum(int x, int y) {
	return x + y;
}

double norm(double* x, int n) {
	double sum = 0;
	for(int i = 0; i < n; i++)
		sum += x[i]*x[i];
	return sqrt(sum);
}

double func1(double* x, int n) {
	if(n != 2) {
		cout << "input np.array must be of length 2" << endl;
		return 0.0;
	}

	return x[0]*x[0] + x[1]*x[1];
}