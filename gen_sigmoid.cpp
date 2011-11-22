#include <iostream>
#include <cmath>

using namespace std;

double sigmoid(double arg){
    return (double)1 / (1 + exp(-1 * arg));
}

int main(void){
    int alpha = 16;
    int beta = 8;
    int gamma = 16;
    double tmp;

    cout << "/*" << endl; 
    cout << "this constant is generated automatically" << endl;
    cout << "its alpha  = " << alpha << ", beta = " << beta << ", gamma = " << gamma << endl;
    cout << "*/" << endl;

    cout << "int sig[" << pow(2,alpha)+1 << "] = {" << endl;
    for(int i=0;i<pow(2,alpha)+1;i++){
        //cerr << sigmoid((double)i/(1<<alpha) - beta ) << endl;
        tmp = (double)(i * 2 * beta) / (pow(2,alpha) - 1);
        cout << "    " << (int)((sigmoid(tmp - beta)) * (1 << gamma)) << "," << endl;
    }
    cout << "};" <<endl;
}
