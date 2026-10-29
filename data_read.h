#ifndef DATA_READ_H
#define DATA_READ_H

#include<bits/stdc++.h>

std::vector<std::vector<float>> load_mnist_images(const std::string& filepath){
	std::ifstream file(filepath,std::ios::binary);
	if(!file) exit(1);
	long long magic,num,rows,cols;
	file.read(reinterpret_cast<char*>(&magic),4);
	file.read(reinterpret_cast<char*>(&num),4);
	file.read(reinterpret_cast<char*>(&rows),4);
	file.read(reinterpret_cast<char*>(&cols),4);
	
	magic = __builtin_bswap32(magic);
	num = __builtin_bswap32(num);
	rows = __builtin_bswap32(rows);
	cols = __builtin_bswap32(cols);
	
	std::vector<std::vector<float>> images(num);
	for(long long i=0;i<num;i++){
		images[i].resize(rows*cols);
		for(int j=0;j<rows*cols;j++){
			int pixel;
			file.read(reinterpret_cast<char*>(&pixel),1);

			images[i][j]=pixel/255;//πÈ“ªªØ
		}
	}
	return images;
}

std::vector<int> load_mnist_labels(const std::string& filepath){
	std::ifstream file(filepath,std::ios::binary);
	long int magic,num;
	file.read(reinterpret_cast<char*>(&magic),4);
	file.read(reinterpret_cast<char*>(&num),4);
	magic = __builtin_bswap32(magic);
	num = __builtin_bswap32(num);
	std::vector<int> labels(num);
 	for(long int i=0;i<num;i++){
	 	 file.read(reinterpret_cast<char*>(&labels[i]),1);	
	}
	return labels;
}





#endif 
