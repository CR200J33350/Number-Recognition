#ifndef MODEL_H
#define MODEL_H

#include "matrix.h"
#include<bits/stdc++.h>

class BaseModel{
	// 神经网络基础架构，以此为基础实现
	public:
		virtual ~BaseModel() = default;
		virtual int forward(const std::vector<float> input) = 0;
		virtual void train(const std::string& bin_file,int epoch) = 0;
		virtual bool load_weights(const std::string& filepath) = 0;
		virtual bool save_weights(const std::string& filepath) = 0;
};

class recognize_model: BaseModel{
	private:
		matrix<float> w1,b1;
		matrix<float> w2,b2;
	public: 
		recognize_model(int input_size, int hidden_size,int output_size):
			w1(hidden_size,input_size), b1(hidden_size,1),
			w2(output_size,hidden_size),b2(output_size,1) {}
		int forward(const std::vector<float>& input){
			if(input.size() != w1.c){
				throw std::invalid_argument("输入的啥奶龙玩意");
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
		void train(const std::string& bin_file,int epoch) override{
			matrix<float> final_output;
		    matrix<float> hidden_output;
		    matrix<float> output_grad;
		    matrix<float> hidden_grad;
		}
};



#endif
