#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <bits/stdc++.h>
#include <netinet>
#include "model.h"
using namespace std;

#define PORT 8888
#define INPUT_SIZE 784

int main(){
	int server_fd = socket(AF_INET,SOCK_STREAM,0);
	if(server_fd<0) return -1;
	struct sockaddr_in addr;
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = INADDR_ANY;
	addr.sin_port = htons(PORT);
	if(bind(server_df,(struct sockaddr*)&addr,sizeof(addr))<0){
		return -1;
	}
	listen(server_fd,3);
	cout<<"服务端启动，监听窗口为"<<PORT<<"\n";
	recognize_model model(784,128,10);
	model.load_weights("weights.bin");
	while(1){
		struct sockaddr_in client_addr;
		socklen_t len = sizeof(client_addr);
		int client_fd = accept(server_fd,(struct sockaddr*)&client_addr, &len);
		if(client_fd < 0) continue;
		vector<float> input(INPUT_SIZE);
		size_t total = 0;
		while(total < INPUT_SIZE*sizeof(float) && n > 0){
			ssize_t n = recv(cliend_fd,reinterpret_cast<char*>(input,data())+total,INPUT_SIZE*sizeof(float)-total,0);
		    total += n
		}
		int pred = model.forward(input);
		cout<<"你写的最有可能是："<<pred<<"\n";
		send(client_fd,&pred,sizeof(int),0);
		close(client_fd);
	}
	close(server_fd);
	return 0;
}
