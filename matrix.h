#ifndef _MATRIX_H_
#define _MATRIX_H_

#include<vector>
#include<stdexcept>
#include<cmath>
#include<omp.h>

template <typename T>

class matrix{
	public:
		int r,c; // row 行 column 列
		std::vector<T> data;
		
		matrix(int r,int c,T val = T(0)):r(r),c(c),data(r*c,val){}
		// 声明矩阵：matrix(i,j) 表示第r 行 c 列的元素, data 表示所有数值，其中用vector 存储数值。
		
		// 只读操作用const保证对象状态不会被修改
		T& operator()(int i,int j)const{
			return data[i*c+j];
		}
		bool is_same_shape(matrix<T> a){
			return c == a.c&&r == a.r;
		}
		// 乘法操作，行乘列，使用多线程优化
		matrix<T> multiply(const matrix<T>& other){
			if(c != other.r) throw std::invalid_argument("第一个矩阵行数不等于第二个矩阵的列数");
			matrix<T> resl(r,other.c);
			omp_set_num_threads(4);
			#pragma omp parallel for
			for(int i=0;i<r;i++){
				for(int j=0;j<other.c;j++){
					T sum = T(0);
					for(int k=0;k<c;k++){
						sum += (*this)(i,k)*other(k,j);
					}
					resl(i,j) = sum;
				}
			}
			return resl;
		}
		// 加法操作，对应元素相加
		matrix<T>& add(const matrix<T>& other){
			if(!is_same_shape(other)){
				throw std::invalid_argument("矩阵不同型，无法相加");
			}
			for(int i=0;i< data.size();i++){
				data[i] += other.data[i];
			}
			return *this;
			// 操作的是目前这个矩阵，返回本身，*this 类似Python 中的 self 
		}
		// 转置操作
		matrix<T>& trasp() const{
			matrix resl(c,r);
			for(int i=0;i<r;i++){
				for(int j=0;j<c;j++){
					resl(j,i) = (*this)(i,j);
				}
			}
			return resl;
		}
		// 用激活函数映射矩阵
		matrix<T> relu(){
			for(auto& i:data){
				if(i < T(0)) i = T(0);
			}
			return *this;
		}
		// 将矩阵输入SoftMax函数中
		matrix<T> smx() const{
			matrix<T> resl(r,c);
			T maximum  = data[0];
			for(auto v:data){
				if(v > maximum) maximum = v;
			}
			
			T sum = T(0);
			for(int i=0;i<r;i++){
				// 归一化
				resl(i,0) = exp(data[i]-maximum);
				sum += exp(data[i]-maximum);
			}
			for(int i=0;i<r;i++){
				resl(i,0) /= sum;
			}
			return resl;
		}
		
	
	
};

#endif
