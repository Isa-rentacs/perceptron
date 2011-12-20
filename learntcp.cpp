#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <cstring>
#include <fstream>
#include <sstream>
#include "perceptron.hpp"

using namespace std;

#ifndef LL
#define LL long long
#endif

int parseInt(char *arg){
    string str;
    int ret;

    for(int i=0;i<(int)strlen(arg);i++){
        str += arg[i];
    }

    istringstream iss(str);
    iss >> ret;

    return ret;
}

int main(int argc, char *argv[]){
    vector<string> teacher;
    vector<string> data;
    string str;
    int t=0;
    long long tsum=0;
    int n;
    int ans;
    int takenAll=0, takenNormal=0;
    int gaveup=0, random=0;
    int istaken;
    LL result,e;
    int pred;
    Perceptron p;

    if(argc == 1){
        cout << "input number of teacher. Exitting." <<endl;
        exit(0);
    }
    n = parseInt(argv[1]);
    cout << "history length = " << n << endl;

    //データの読み込み
    ifstream fin("./data.dat");

    while(getline(fin,str)){
        data.push_back(str);
    }

    cout << "data loaded. # of data =" << data.size() << endl;
  
    //n番目から予測を始める
    for(int x=0;x<100;x++){
    for(int i=n;i<(int)data.size();i++){
        teacher.clear(); //教師データのクリア
        p.init();        //perceptron内parameterのクリア
        for(int j=0;j<n;j++){
            teacher.push_back(data[i-1-j]); //教師データを加えていく
        }
        //与えた教師データでの学習を行う
        e = p.learn(teacher);

        //教師データ(正答)を取得
        istringstream iss(data[i]);
        for(int j=0;j<3;j++){
            iss >> ans;
        }
        iss >> ans;
    
        //paramter出力
        //p.print_param();
        //教師データを流してperceptronからの出力を得る
        result = p.get(data[i]);

        //printf("pred = %llu\n", result);
        //predictonの値を決める
        if(result > (1 << (GAMMA-1))){
            pred = 1;
        }else{
            pred = 0;
        }

        //cout << pred << endl;
        if(pred == ans){
            takenAll++;
            //if(t != LOOP_MAX) takenNormal++;
            istaken = 1;
            //cout << takenAll << endl;
        }else{
            istaken = 0;
        }
        printf("[%d]answer = %d, prediction = %d, istaken=%d, repeated %d times\n",
               i,ans,(int)round(pred),istaken, t);

    }
    }

    printf("taken rate(ALL)=%lf(%d/%d), ",(double)takenAll / ((data.size()- n)*100), takenAll, (int)(data.size() - n)*1000);
}
