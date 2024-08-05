#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm> // std::random_shuffle
#include <cstdlib> // std::rand and std::srand
#include <ctime>    // std::time
#include <opencv2/opencv.hpp>
#include <random>

using namespace std;
using namespace cv;

class LinearRegression {

public:
    double weights;
    double bias;
    double learning_rate;
    vector<double> x_data;
    vector<double> y_data;

    LinearRegression(vector<double> _x_data, vector<double> _y_data, double _weights , double _bias, double _learning_rate) 
        : x_data(_x_data), y_data(_y_data), weights(_weights), bias(_bias), learning_rate(_learning_rate) {}


    double linear_func(double _x_data, double _weights, double _bias) {
        return _x_data * _weights + _bias;
    }

    double loss_func(double _x, double _y, double _weights, double _bias) {
        return pow((linear_func(_x, _weights, _bias) - _y), 2);
    }

    double gradient_update(string grad_name, int idx) {
        double h = 1e-4;  
        if (grad_name == "weights") {
            double forward = loss_func(x_data[idx], y_data[idx], weights + h, bias);
            double backward = loss_func(x_data[idx], y_data[idx], weights - h, bias);
            return (forward - backward) / (2 * h);
        } else {
            double forward = loss_func(x_data[idx], y_data[idx], weights, bias + h);
            double backward = loss_func(x_data[idx], y_data[idx], weights, bias - h);
            return (forward - backward) / (2 * h);
        }
    }

    // Update weights and bias using the shuffled indices
    vector<double> update_by_suffled_indices() {
        // Create an index vector and shuffle it
        vector<int> indices(x_data.size());
        for (int i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }

        shuffle(indices.begin(), indices.end(), std::default_random_engine(std::time(0)));

        // Update weights and bias using the shuffled indices
        for (int i : indices) {
            double grad_w = gradient_update("weights", i);
            double grad_b = gradient_update("bias", i);

            weights = weights - learning_rate * grad_w;
            bias = bias - learning_rate * grad_b;
        }

        vector<double> update_wb(2);
        update_wb[0] = weights;
        update_wb[1] = bias;

        return update_wb;
    }
    
    // Update weights and bias using the sequential indices
    vector<double> update_by_sequential_indices() {
        for (int i = 0; i < x_data.size(); ++i) {
            double grad_w = gradient_update("weights", i);
            double grad_b = gradient_update("bias", i);

            weights = weights - learning_rate * grad_w;
            bias = bias - learning_rate * grad_b;
        }

        vector<double> update_wb(2);
        update_wb[0] = weights;
        update_wb[1] = bias;

        return update_wb;
    }
};

class RGBImage {
    public:
        static void convertImageToVector(Mat& image, vector<int>& data) {
            data.clear();
            for(int y = 0; y < image.rows; y++)
            {
                for(int x = 0; x < image.cols; x++)
                {
                    Vec3b color = image.at<Vec3b>(Point(x, y));
                    data.push_back(color[0]); // B channel
                    data.push_back(color[1]); // G channel
                    data.push_back(color[2]); // R channel
                }
            }
        }

        static void convertVectorToImage(vector<int>&data, Mat& image) {
            int index = 0;
            for(int y = 0; y < image.rows; y++)
            {
                for(int x = 0; x < image.cols; x++)
                {
                    Vec3b& color = image.at<Vec3b>(Point(x, y));
                    color[0] = data[index++]; // B channel
                    color[1] = data[index++]; // G channel
                    color[2] = data[index++]; // R channel
                }
            }
        }
};

int main()
{
    
    vector<double> x_data = {1, 2, 3, 4, 5};
    vector<double> y_data = {2, 4, 6, 8, 10};
    
    double weights = ((double)rand() / RAND_MAX);
    double bias = ((double)rand() / RAND_MAX);
    double learning_rate = 0.01;

    LinearRegression lr(x_data, y_data, weights, bias, learning_rate);
    
    for(int epoch = 0; epoch < 2000; epoch++)
    {
        cout << "epoch : " << epoch << " " << "weights : " << "[" <<  weights << "]" << " " << "bias : " << "[" <<  bias << "]" ;
        cout << "\n";
        vector<double> update_data = lr.update_by_suffled_indices();
        weights = update_data[0];
        bias = update_data[1];
    }

    cout << "Updated Weights : " <<  weights << endl;
    cout << "Updated bias : " <<  bias << endl;

    return 0;
}