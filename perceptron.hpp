#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

using namespace std;

#define GAIN 1
#define L 3
#define M 4
#define N 1
#define eta 0.1

class Perceptron{
public:
  Perceptron(){
    debug = false;

    srand(time(NULL));
    for(int i=0;i<L+1;i++){
      for(int j=0;j<M;j++){
	wlm[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
	if(debug) cout << wlm[i][j] << endl;
	dlm[i][j] = 0;
      }
    }
    for(int i=0;i<M+1;i++){
      for(int j=0;j<N;j++){
	wmn[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
	if(debug) cout << wmn[i][j] << endl;
	dmn[i][j] = 0;
      }
    }
  }

  void init(void){
    for(int i=0;i<L+1;i++){
      for(int j=0;j<M;j++){
	wlm[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
	if(debug) cout << wlm[i][j] << endl;
	dlm[i][j] = 0;
      }
    }
    for(int i=0;i<M+1;i++){
      for(int j=0;j<N;j++){
	wmn[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
	if(debug) cout << wmn[i][j] << endl;
	dmn[i][j] = 0;
      }
    }
  }

  void print_param(void){
	for(int i=0;i<L+1;i++){
	  for(int j=0;j<M;j++){
	    printf("wlm[%d][%d] = %lf\n",i,j,wlm[i][j]);
	  }
	}
	for(int i=0;i<M+1;i++){
	  for(int j=0;j<N;j++){
	    printf("wmn[%d][%d] = %lf\n",i,j,wmn[i][j]);
	  }
	}
	return;
  }

  double learn(vector<string> arg){
	int n=arg.size(); //# of teacher data
	int teacher;
	double result;
	double error=0;
	//領域の初期化
	memset(dlm, 0, sizeof(dlm));
	memset(dmn, 0, sizeof(dmn));

	//全てのデータに対する微分係数の和を求めていく
	for(int i=0;i<n;i++){
	  istringstream iss(arg[i]);
	  double delta_k, delta_j;

	  for(int j=0;j<L;j++){
		iss >> teacher;
	  }
	  iss >> teacher;
	  
	  result = get(arg[i]);

	  if(debug) cout << "Nout = " << result << endl;

	  //M->Nの偏微分値
	  delta_k = (teacher - result) * result * (1 - result);
	  for(int j=0;j<M+1;j++){
		for(int k=0;k<N;k++){
		  if(j != M){
			dmn[j][k] += delta_k * Mout[j];
		  }else{
			dmn[j][k] += delta_k * -1;
		  }
		}
	  }
	  
	  
	  //L->Mの偏微分値
	  for(int j=0;j<M;j++){
		delta_j = Mout[j] * (1 - Mout[j]) * delta_k * wmn[j][0];
		for(int k=0;k<L+1;k++){
		  if(k != L){
			dlm[k][j] += delta_j * Lout[k];
		  }else{
			dlm[k][j] += delta_j * -1;
		  }
		}
	  }
	  error += (teacher - result) * (teacher - result);
	}
       
	for(int i=0;i<L+1;i++){
	  for(int j=0;j<M;j++){
	    wlm[i][j] += dlm[i][j] * eta;
	  }
	}
	for(int i=0;i<M+1;i++){
	  for(int j=0;j<N;j++){
	    wmn[i][j] += dmn[i][j] * eta;
	  }
	}
	
	if(debug)cout << "Error:" << error << endl; 
	
	return error*0.5;
  }

  double get(string str){
	istringstream iss(str);
	for(int i=0;i<L;i++){
	  iss >> Lout[i];
	  if(debug) printf("Lout[%d] = %lf\n", i, Lout[i]);
	}
	for(int i=0;i<M;i++){
	  Min[i] = 0;
	  for(int j=0;j<L;j++){
		Min[i] += wlm[j][i] * Lout[j]; 
	  }
	  Min[i] += wlm[L][i] * -1;
	  if(debug) printf("Min[%d] = %lf\n", i, Min[i]);
	}

	for(int i=0;i<M;i++){
	  Mout[i] = (double)1 / (1 + exp(-1 * Min[i]));
	  if(debug) printf("Mout[%d] = %lf\n",i , Mout[i]);
	}

	for(int i=0;i<N;i++){
	  Nin[i] = 0;
	  for(int j=0;j<M;j++){
		Nin[i] += wmn[j][i] * Mout[j];
		if(debug) printf("Nin[%d](%lf) += wmn[%d][%d](%lf) * Mout[%d](%lf)\n", i, Nin[i], j,i,wmn[j][i], i,Mout[j]);
	  }
	  Nin[i] += wmn[M][i] * -1;
	  if(debug) printf("Nin[%d] = %lf\n", i, Nin[i]);
	}

	return (double) 1 / (1 + exp(-1 * Nin[0]));
  }

private:
  double wlm[L+1][M];
  double wmn[M+1][N];
  double dlm[L+1][M];
  double dmn[M+1][N];
  double Lout[L];
  double Min[M];
  double Mout[M];
  double Nin[N];
  bool debug;


  double output;
};
