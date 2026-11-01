#ifndef MODEL_H
#define MODEL_H

#include "matrix.h"
#include "data_read.h"
#include<vector>
#include<stdlib.h>
#include<random>
#include<cmath>
using namespace std;

class recognize_model{
	private:
		matrix<float> w1,b1;
		matrix<float> w2,b2;
	public: 
		recognize_model(int input_size, int hidden_size,int output_size):
			w1(hidden_size,input_size), b1(hidden_size,1),
			w2(output_size,hidden_size),b2(output_size,1) {}
		int forward(const vector<float>& input){
			if(input.size() != w1.c){
				exit(1);
			}
			// 生成一个列向量
			matrix<float> col_input(w1.c,1);
			for(int i=0;i<w1.c;i++){
				col_input(i,0) = input[i];
			}
			// 进行前向传播
			// 线性变换
			matrix<float> z1 = w1.multiply(col_input);
			z1.add(b1);
			// 经过激活函数
			z1.relu();
			// 进行前向传播
			// 线性变换
			matrix<float> z2 = w2.multiply(z1);
			z2.add(b2);
			matrix<float> prob = z2.smx();
			// 经过激活函数
			int prediction = 0;
			float max_p = prob(0,0);
			for(int i=1;i<prob.r;i++){
				if(prob(i,0)>max_p){
					max_p = prob(i,0);
					prediction = i;
				}
			}
			return prediction;
		}
		void init_weights(float a = 0.01f){
			mt19937 gen(42); // 随机生成一堆数字作为初始的权重
			normal_distribution<float> dist(0.0f,a); // 正态分布初始化
			for(auto& v: w1.data) v = dist(gen);
			for(auto& v: w2.data) v = dist(gen); 
		}
		
		void train(const string path, int epochs,float lr = 0.1f){
			// lr 代表学习率
			cout << "图片读取中...\n";
			auto images = load_mnist_images(path + "/train-images.idx3-ubyte");
			cout << "标签读取中...\n";
			auto labels = load_mnist_labels(path + "/train-labels.idx1-ubyte");
			printf("数据已就绪，%d轮训练开始:\n",epochs);
			if(images.size()!=labels.size()) exit(1);
			init_weights();
			size_t n =images.size();
			for(int ep=0;ep<epochs;ep++){
				float total_loss = 0;
				for(int i=0;i<n;i++){
					matrix<float> x(w1.c,1);
					for(int j=0;j<epochs;j++) x(j,0) = images[i][j];
					matrix<float> z1 = w1.multiply(x);
					z1.add(b1);
					matrix<float> a1 = z1;
					a1.relu();
					matrix<float> z2 = w2.multiply(a1);
					z2.add(b2);
					matrix<float> a2 = z2.smx();
					matrix<float> dz2 = a2;
					int label = labels[i];
					dz2(label, 0) -= 1.0f;
					total_loss -= log(a2(label,0) + 1e-8f);
					
					matrix<float> dw2 = dz2.multiply(a1.trasp());
					matrix<float> db2 = dz2;
					
					matrix<float> da1 = w2.trasp().multiply(dz2);
					matrix<float> dz1 = da1;
					for(int j =0;j<dz1.r;j++){
						if(z1(j,0) <= 0) dz1(j,0) = 0;
					}
					matrix<float> dw1 = dz1.multiply(x.trasp());
					matrix<float> db1 = dz1;
					for(size_t j=0; j<w2.data.size();j++) w2.data[j] -= lr*dw2.data[j];
					for(size_t j=0; j<b2.data.size();j++) b2.data[j] -= lr*db2.data[j];
					for(size_t j=0; j<w1.data.size();j++) w1.data[j] -= lr*dw1.data[j];
					for(size_t j=0; j<b1.data.size();j++) b1.data[j] -= lr*db1.data[j];
				    }
				printf("Epoch %d, Avg Loss: %.4f\n", ep, total_loss);
				}
			printf("模型训练已完成\n");
			return;
		}
		void load_weights(string filepath){
			ifstream in(filepath,ios::binary);
			in.read(reinterpret_cast<char*>(w1.data.data()),w1.data.size()*sizeof(float));
			in.read(reinterpret_cast<char*>(b1.data.data()),b1.data.size()*sizeof(float));
			in.read(reinterpret_cast<char*>(w2.data.data()),w2.data.size()*sizeof(float));
			in.read(reinterpret_cast<char*>(b2.data.data()),b2.data.size()*sizeof(float));
		}
		void save_weights(std::string filepath){
			ofstream out(filepath,ios::binary);
			out.write(reinterpret_cast<char*>(w1.data.data()),w1.data.size()*sizeof(float));
			out.write(reinterpret_cast<char*>(b1.data.data()),b1.data.size()*sizeof(float));
			out.write(reinterpret_cast<char*>(w2.data.data()),w2.data.size()*sizeof(float));
			out.write(reinterpret_cast<char*>(b2.data.data()),b2.data.size()*sizeof(float));
		}
};



#endif
