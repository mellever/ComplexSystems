#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <string>
#include <cstdlib>
#include <numeric>
#include <complex>
#include <algorithm>
#include <omp.h>
#include <iterator>

using namespace std;

// Constants
int N = 2000; //2000
int T = 200; //200
int Kmax = 10;
int k = 6;
float K;
float dK = 0.2;
float dt = 0.1;
float pmax = 1;
float dp = 0.05;
unsigned int seed = 46;
int Ksteps = Kmax/dK+1;
int tsteps = T/dt;
int psteps = pmax/dp+1;


// For random generation
mt19937 urbg {seed};  

vector<float> random_uniform(){
	vector<float> theta0;
	uniform_real_distribution<float> distr2 {0, 2*M_PI};
	for (int i=0; i<N; i++){
		auto const random = distr2(urbg);
		theta0.push_back(random);
	}
	return theta0;
}

vector<float> random_normal(){
	vector<float> w0;
	uniform_real_distribution<float> distr2 {0, 1};
	for (int i=0; i<N; i++){
		auto norm = normal_distribution<double>{0, 1};
		auto value = norm(urbg);
		w0.push_back(value);
	}
	return w0;
}

// Function to generate a Watts-Strogatz graph and return its adjacency matrix
vector<vector<int>> generate_adjacency(int num_nodes, int k, double beta, unsigned int seed) {
    // Create a random number generator
    mt19937 rng(seed);

    // Create a ring lattice
    vector<vector<int>> adjacency_matrix(num_nodes, vector<int>(num_nodes, 0));
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 1; j <= k / 2; ++j) {
            int target = (i + j) % num_nodes;
            adjacency_matrix[i][target] = 1;
            adjacency_matrix[target][i] = 1; // Undirected graph, so we set the reverse edge
        }
    }

    // Rewire edges with probability beta
    uniform_real_distribution<double> rand_prob(0.0, 1.0);
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 1; j <= k / 2; j++) {
            if (rand_prob(rng) < beta) {
                int target = (i + j) % num_nodes;

                // Remove the current edge
                adjacency_matrix[i][target] = 0;
                adjacency_matrix[target][i] = 0;

                // Randomly rewire to a different node
                int new_target;
                do {
                    new_target = rng() % num_nodes;
                } while (new_target == i || adjacency_matrix[i][new_target] == 1);

                // Add the new edge
                adjacency_matrix[i][new_target] = 1;
                adjacency_matrix[new_target][i] = 1;
            }
        }
    }

    return adjacency_matrix;
}


int main() {
	vector<float> w = random_normal();
	vector<float> theta = random_uniform();
	vector<vector<float>> data(psteps, vector<float> (Ksteps));

	for (int m=0; m<psteps; m++){
		float p = m*dp;
		
		// Generate a Watts-Strogatz graph and get its adjacency matrix
  	vector<vector<int>> A = generate_adjacency(N, k, p, seed);

		for (int l=0; l<Ksteps; l++){
			K = dK*l;

	  	#pragma omp parallel for
			for (int t=0; t<tsteps; t++){
				for (int i=0; i<N; i++){
					float s=0;
					for (int j=0; j<N; j++){
						s += A[i][j]*sin(theta[j]-theta[i]);
					}
					theta[i] += dt*(w[i] + K*s/k);
				}
			}
			float sum_t = 0;
			for (int i=0; i<N; i++){
				sum_t += theta[i];
			}
			float mean_t = sum_t/N;

			complex<float> sum = 0;
			for (int i=0; i<N; i++){
				sum+= exp(complex<float>(0,1) * (theta[i]-mean_t));
			}
			float r = abs(sum)/N;
			cout << "p = " << p << ", K = "  << K << ", r = " << r << endl;
			data[m][l] = r;
		}
	}
	ofstream output_file("data.txt");
    for (int i = 0; i < data.size(); ++i) {
        for (auto it = data[i].begin(); it != data[i].end(); ++it) {
            output_file << *it << "\t";
        }
        output_file << endl;
    }
    output_file.close();
}


