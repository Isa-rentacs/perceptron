#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include "./sigmoid.hpp"
#include "./pow2.hpp"

using namespace std;

#define GAIN 1
#define L 3
#define M 4
#define N 1
#define eta 0.1
#define ALPHA 16
#define BETA 6
#define GAMMA 16
#define DELTA 16
#define LOOP_MAX 100
#define LL long long

class Perceptron{
public:
    Perceptron(){
        srand(time(NULL));
        init();
    }

    void init(void){
        /*
         * 各edgeの重みを設定。初期値は[-1,1]->
         * 変換して[-DELTA, DELTA]にする
         */
        for(int i=0;i<L+1;i++){
            for(int j=0;j<M;j++){
                wlm[i][j] = rand() % (pow2[DELTA+1]+1) - pow2[DELTA];
                if(is_debug) printf("[perceptron::init]wlm[i][j] = %lld\n", wlm[i][j]);
                dlm[i][j] = 0;
            }
        }
        for(int i=0;i<M+1;i++){
            for(int j=0;j<N;j++){
                wmn[i][j] = rand() % (pow2[DELTA+1]+1) - pow2[DELTA];
                if(is_debug) printf("[perceptron::init]wmn[i][j] = %lld\n", wmn[i][j]);
                dmn[i][j] = 0;
            }
        }
    }

    void print_param(void){
        for(int i=0;i<L+1;i++){
            for(int j=0;j<M;j++){
                printf("[perceptron::print_param]wlm[%d][%d] = %lld\n",i,j,wlm[i][j]);
            }
        }
        for(int i=0;i<M+1;i++){
            for(int j=0;j<N;j++){
                printf("[perceptron::print_param]wmn[%d][%d] = %lld\n",i,j,wmn[i][j]);
            }
        }
        return;
    }

    LL learn(vector<string> arg){
        int n=arg.size(); //# of teacher data
        int teacher;
        LL result;
        LL error=0;
        //領域の初期化
        memset(dlm, 0, sizeof(dlm));
        memset(dmn, 0, sizeof(dmn));

        //全てのデータに対する微分係数の和を求めていく
        //一括修正法:それぞれの教師データの誤差の和を
        //小さくする方向に修正する
        for(int x=0;x<LOOP_MAX;x++){

            //変化量領域の初期化
            memset(dlm, 0, sizeof(dlm));
            memset(dmn, 0, sizeof(dmn));

            for(int i=0;i<n;i++){
                istringstream iss(arg[i]);
                LL delta_k, delta_j;

                for(int j=0;j<L;j++){
                    iss >> teacher;
                }
                iss >> teacher;
	  
                //実際の値の2^GAMMA倍の値が出てくる
                result = get(arg[i]);

                //M->Nの偏微分値
                //最終的な値は本来の値の2^DELTA倍になっていればよい
                delta_k = (teacher << GAMMA) - result;
                delta_k *= (1 << GAMMA) - result;
                delta_k >>= GAMMA;
                delta_k *= result;
                delta_k >>= GAMMA;
                for(int j=0;j<M+1;j++){
                    for(int k=0;k<N;k++){
                        if(j != M){
                            dmn[j][k] += (((delta_k * Mout[j]) >> GAMMA) << DELTA) >> GAMMA;
                        }else{
                            dmn[j][k] += ((delta_k * -1) << DELTA) >> GAMMA;
                        }
                    }
                }
	  
	  
                //L->Mの偏微分値
                for(int j=0;j<M;j++){
                    delta_j = (delta_k * wmn[j][0]) >> DELTA;
                    delta_j *= Mout[j];
                    delta_j >>= GAMMA;
                    delta_j *= (1<<GAMMA) - Mout[j];
                    delta_j >>= GAMMA;
                    for(int k=0;k<L+1;k++){
                        if(k != L){
                            dlm[k][j] += (((delta_j * Lout[k]) >> GAMMA) << DELTA) >> GAMMA;
                        }else{
                            dlm[k][j] += ((delta_j * -1) << DELTA) >> GAMMA;
                        }
                    }
                }
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
        }
        
        if(is_debug) cout << "Error:" << error << endl; 

        return error;
    }


    /*
     * 入力に対する予測を出力する
     * input  : values in form of string (L values)
     * output : prediction
     */
    LL get(string str){
        istringstream iss(str);
        //L層の出力として入力データを読み込む
        for(int i=0;i<L;i++){
            iss >> Lout[i];
        }
        //M層のi番目ノードに対する入力値を計算する
        for(int i=0;i<M;i++){
            Min[i] = 0;
            //L層の出力*対応するedgeの重みの和を計算
            for(int j=0;j<L;j++){
                Min[i] += wlm[j][i] * Lout[j]; 
            }
            //M層のi番目ノードの閾値分を入力から減算する
            //wlm[L][i]: 閾値θ
            Min[i] += wlm[L][i] * -1;
            if(is_debug) printf("[perceptron::get]Layer-M,%d-th Node, RowInputSum = %lld\n", i, Min[i]);
        }

        //M層i番目のノードの出力値を計算する
        LL ModifiedInput;
        for(int i=0;i<M;i++){
            ModifiedInput = (Min[i] >> (1 + DELTA - ALPHA)) / BETA + pow2[ALPHA-1];
            if(is_debug) printf("[perceptron::get]Layer-M,%d-th Node, ModifiedInputSum = %lld\n", i, ModifiedInput);
            if(0 <= ModifiedInput && ModifiedInput < (1 << ALPHA)){
                Mout[i] = sigmoid[ModifiedInput];
            }else if(ModifiedInput < 0){
                Mout[i] = 0;
            }else{
                Mout[i] = 1 << GAMMA;
            }
            if(is_debug) printf("[perceptron::get]Layer-M,%d-th Node, ModifiedOutput = %lld\n",i , Mout[i]);
        }

        //N層i番目のノードへの入力値を計算する
        for(int i=0;i<N;i++){
            Nin[i] = 0;
            for(int j=0;j<M;j++){
                //M層の出力*対応するedgeの重みの和を計算する
                Nin[i] += wmn[j][i] * Mout[j];
                //if(is_debug) printf("Nin[%d](%lld) += wmn[%d][%d](%lld) * Mout[%d](%lld)\n", i, Nin[i], j,i,wmn[j][i], i,Mout[j]);
            }
            Nin[i] += wmn[M][i] * -1;
            if(is_debug) printf("[perceptron::get]Layer-N,%d-th Node, RowInputSum = %lld\n", i, Min[i]);
            //if(is_debug) printf("Nin[%d] = %lf\n", i, Nin[i]);
        }

        ModifiedInput = (Nin[0] >> (1 + GAMMA + DELTA - ALPHA)) / BETA + pow2[ALPHA-1];
        if(is_debug) printf("[perceptron::get][Layer-N,0-thNode,ModifiedInputSum] = %lld\n", ModifiedInput);
        if(is_debug) printf("[perceptron::get][Layer-N,0-thNode,ModifiedOutput] = %lld\n", sigmoid[ModifiedInput]);
        if(0 <= ModifiedInput && ModifiedInput < (1 << ALPHA)){
            return sigmoid[ModifiedInput];
        }else if(ModifiedInput < 0){
            return 0;
        }else{
            return 1 << GAMMA;
        }
        return sigmoid[ModifiedInput];
    }

private:
    LL wlm[L+1][M];
    LL wmn[M+1][N];
    LL dlm[L+1][M];
    LL dmn[M+1][N];
    LL Lout[L];
    LL Min[M];
    LL Mout[M];
    LL Nin[N];
    const static bool is_debug = false;


    LL output;
};
