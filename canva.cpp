#include<opencv2/opencv.hpp>
#include<iostream>
#include "model.h"

using namespace std;

cv::Mat canvas(280,280,CV_8UC1,cv::Scalar(0));
bool drawing = false;


void draw(int event, int x, int y, int flags, void* param) {
	if (event == cv::EVENT_LBUTTONDOWN) drawing = 1;
	else if (event == cv::EVENT_LBUTTONUP) drawing = 0;
	// 检测鼠标
	if (event == cv::EVENT_MOUSEMOVE && drawing) {
		cv::circle(canvas, cv::Point(x, y), 15, 255, -1);
	}
}

vector<float> guiyihua(cv::Mat img) {
	cv::Mat resized;
	cv::resize(img, resized, cv::Size(28, 28), 0, 0, cv::INTER_AREA);
	cv::Mat floatMat;
	resized.convertTo(floatMat, CV_32FC1, 1.0 / 255.0);
	vector<float> flat(784);
	int count = 0;
	for (int r = 0; r < 28; r++) {
		for (int c = 0; c < 28; c++) {
			flat[count] = floatMat.at<float>(r, c);
		}
	}
	return flat;
}

int main() {
	recognize_model model(784, 128, 10);
	model.init_weights();
	model.load_weights("weights.bin");
	cv::namedWindow("recognize",cv::WINDOW_AUTOSIZE);
	cv::moveWindow("recognize", 0, 0);
	cv::setMouseCallback("recognize", draw);
	while (true) {
		cv::imshow("recognize", canvas);
		int key = cv::waitKey(1) & 0xFF;
		if (key == 13) {
			vector<float> input = guiyihua(canvas);
			int pred = model.forward(input);
			std::string title = "result:" + to_string(pred);
			printf("result：%d \n", pred);
;
		}
		else if (cv::getWindowProperty("recognize", cv::WND_PROP_VISIBLE) < 1) {
			break;
		}
		else if (key == 'c') {
			canvas.setTo(0);
		}
		else if (key == 27) {
			break;
		}
	}
	return 0;
}
