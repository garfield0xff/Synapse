#include<iostream>
#include<algorithm>
#include<vector>
#include<cmath>
#include<random>
using namespace std;
const int MX = 300;
vector<int>v;

// wx + bias;
double linear_func(double x_data,double weights, double bias){
    return x_data*weights+bias;
}

// random linear - 
vector<double> loss_func(double weights,double bias,vector<int>x_data,vector<int>y_data){
    vector<double>grad(x_data.size(),0);
    for(int i=0;i<x_data.size();i++){
        grad[i] = 1/(i+1)*pow( (linear_func(x_data[i],weights,bias) - y_data[i]),2);
    }

    return grad;
}

// 미분의 중앙차분으로 구현 함, (전진차분, 후진 차분 도 있음)
double gradient_update(vector<int>x_data, vector<int>y_data, double weights, double bias, string grad_name){
    double h = log(10)-4;
    if(grad_name == "weights"){
        vector<double> forward = loss_func(weights + h, bias, x_data,y_data);
        vector<double> backward = loss_func(weights - h, bias, x_data,y_data);
        double forward_sum = 0;
        double backward_sum = 0;

        for(int i=0;i<forward.size();i++){
            forward_sum += forward[i];
            backward_sum += backward[i];
        }
        return (forward_sum - backward_sum)/(2*h);
    }else{
        vector<double> forward = loss_func(weights , bias + h, x_data,y_data);
        vector<double> backward = loss_func(weights , bias - h, x_data,y_data);
        double forward_sum = 0;
        double backward_sum = 0;

        for(int i=0;i<forward.size();i++){
            forward_sum += forward[i];
            backward_sum += backward[i];
        }
        return (forward_sum - backward_sum)/(2*h);
    }
}

vector<double> update(double learning_rate, double grad_w,double grad_b, double weights,double bias){
    vector<double> update_wb(2);
    weights = weights - learning_rate*grad_w;
    bias = bias - learning_rate*grad_b;

    update_wb[0] = weights;
    update_wb[1] = bias;

    return update_wb;
}

int main(){
    ios_base::sync_with_stdio(0);
    cin.tie(0);

    // y = x + 1, w = 1, b = 1
    // opencv 이용해서 이미지를 가지고 뭘 해보고 싶었지만
    // 무지이슈.. 윈도우 이슈 희진님이 보내주신거 보고 opencv는 다음에 해보기로
    // 영상 몇개 좀더 찾아보고 하기로 하고 일단 pass 
    
    // 어떤 빵 가게에 하루당 소보루빵 제품의 판매량 데이터의 관계를 나타내 보고자 한다.
    // 날짜, 시간, 일차
    // vector<int>x_data;
    // for(int i=0;i<30;i++){
    //     x_data.push_back(i+1);
    // }
    // 판매량 (빵 갯수)
    // vector<int>y_data={25,20,21,18,21,15,19,24,20,16,18,22,13,19,24,13,18,29,24,17,12,11,23,5,27,14,15,31,21,9};

    // 숫자들이 너무 이상해서 x데이터 y데이터를 바꿔서 진행도 해봄
    // 출력 당시 
    //updated Weight : 8.54781 ??? 
    //updated bias   : 9.11015 ???
    // 위에 숫자들은 뭘 의미하는 거지?

    // 날짜, 시간, 일차
    vector<int>y_data;
    for(int i=0;i<30;i++){
        y_data.push_back(i+1);
    }

    // 판매량 (빵 갯수)
    vector<int>x_data={25,20,21,18,21,15,19,24,20,16,18,22,13,19,24,13,18,29,24,17,12,11,23,5,27,14,15,31,21,9};

    // 출력 당시
    //Updated Weights : 0.0174307
    //Updated bias : 0.564232
    
    double weights = ((double)rand()/ RAND_MAX);
    double bias = ((double)rand()/ RAND_MAX);

    cout << "weight: " << weights << '\n';
    cout << "bias: " << bias << '\n';

    for(int i=0;i<MX;i++){
        cout << "weight : " << "[" << weights << "]  " << "bias : [" << bias << "]\n";
        double grad_w = gradient_update(x_data,y_data,weights,bias,"weights");
        double grad_b = gradient_update(x_data,y_data,weights,bias,"bias");
        vector<double> update_data = update(0.001,grad_w,grad_b,weights,bias);
        weights=update_data[0];
        bias=update_data[1];
    }

    cout << "Updated Weights : " << weights << '\n';
    cout << "Updated bias : " << bias << '\n';
    
    return 0;
}