#include<bits/stdc++.h>
#include "model.h"

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10

using namespace std;
int main(){
	recognize_model model(INPUT_SIZE,HIDDEN_SIZE,OUTPUT_SIZE);
	string data_path = "./mnist_data";
	int epochs = 100;
	cout<<"开始训练模型\n";
	model.train(data_path,epochs,0.1f);
	model.save_weights("weights.bin");
	cout<<"模型训练已完成。"
	return 0;
}

