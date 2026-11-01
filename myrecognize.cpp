#include<iostream>
#include "model.h"
#include<stdlib.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10

using namespace std;

int main(){
	int epochs = 90;
	recognize_model model(INPUT_SIZE,HIDDEN_SIZE,OUTPUT_SIZE);
	string data_path = "E:/development/handwritten_number_recognize/myhandwritten_number_recognize/mnist_data/";
	cout<<"开始训练模型\n";
	model.train(data_path,epochs,0.1f);
	model.save_weights("weights.bin");
	cout << "模型训练已完成。";
	system("pause");
	return 0;
}
