#include "cv5/opencv/build/include/opencv.hpp"
#include<bits/stdc++.h>
#include "client.cpp"
using namespace std;
cv::Mat canvas(280,280,CV_8UC1,cv::Scalar(0));
bool drawing = false;

void draw(int event,int x,int y,int flags){
	if(event = cv::EVENT_LBUTTONDOWN) drawing = 1;
	if(event = cv::EVENT_LBUTTONUP) drawing = 0;
	// 检测鼠标
	if(drawing && flags & cv::EVENT_FLAG_LBUTTON){
		cv::circle(canvas,cv::point(x,y),15,cv::SCALAR(255),-1);
	}
}

vector<float> guiyihua(cv::MAT& img){
	cv::Mat resized;
	cv::resize(img, resized, Size(28, 28), 0, 0);
	resized.convertTo(resized,CV_32FC1,1.0/255.0);
	vector<float> flat(784);
	memcpy(flat.data(),resized.data(),784*sizeof(float));
	return flat;
}

int main(){
	cv::namedWindow("手写数字识别程序1.0 按下ENTER进行预测， C清除画布");
	cv::setMouseCallback("请输入数字",on_mouse);
	while(1){
		cv.imshow("画出的数字",canvas);
		int key = cv::waitKey(1) & 0xFF;
		if(key == 13){
			vector<float> input = guiyihua(canvas);
			int pred = send(input);
			string title = "识别结果为:" + to_string(pred);
			printf("  识别结果如下：%d \n",pred);
			cv::setWindowTitle("手写数字识别程序1.0",title)
		}
		else if(key == 'c'){
			canvas = cv::Scalar(0);
		}
		else if(key == 27){
			break ;
		}
	}
	return 0;
}
