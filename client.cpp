#include <sys/socket.h>
#include <bits/stdc++.h>
#include <unistd.h>
#include <arpa/inet.h>
using namespace std;

#define SERVER_IP "169.254.211.93"
#define PORT 8888

int send(vector<float>& image_data){
	int sock = socket(AF_INET,SOCK_STREAM,0);
	struct sockaddr_in serv_addr;
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_port = htons(PORT);
	inet_pton(AF_INET,SERVER_IP,&serv_addr.sin_addr);
	connect(sock,image_data.data(),image_data.size()*sizeof(float),0);
	
	int resl;
	recv(sock,&resl,sizeof(int),0);
	close(sock);
	return resl;
}
