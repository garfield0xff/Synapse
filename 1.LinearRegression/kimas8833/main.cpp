#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/*
    This Linear Regression Code is designed for understand linear relationship between images.
*/

class LinearRegression {

public:
    double weights;
    double bias;
    double learning_rate = 0.0001;
    vector<int> x_data;
    vector<int> y_data;


public:
    // wx + b
    double linear_func(double _x_data, double _weights, double _bias) {
        return _x_data * _weights + _bias;
    }

    // MSE loss function
    vector<double> loss_func(double _weights, double _bias)
    {
        weights = _weights;
        bias = _bias;

        vector<double> grad(x_data.size(), 0);
        for (int idx = 0; idx < x_data.size(); ++idx)
        {
            grad[idx] = 1 / (idx + 1) * pow((linear_func(x_data[idx], weights, bias) - y_data[idx]), 2);
        }
        return grad;
    }

    // update gradient ( central difference )
    double gradient_update(string grad_name)
    {
        double h = log(10) - 4;
        if (grad_name == "weights")
        {
            vector<double> forward = loss_func(weights + h, bias);
            vector<double> backward = loss_func(weights - h, bias);
            double forward_sum = 0;
            double backward_sum = 0;
            for (int i = 0; i < forward.size(); ++i)
            {
                forward_sum += forward[i];
                backward_sum += backward[i];
            }
            return (forward_sum - backward_sum) / (2 * h);
        }
        else {
            vector<double> forward = loss_func(weights, bias + h);
            vector<double> backward = loss_func(weights, bias - h);
            double forward_sum = 0;
            double backward_sum = 0;
            for (int i = 0; i < forward.size(); ++i)
            {
                forward_sum += forward[i];
                backward_sum += backward[i];
            }
            return (forward_sum - backward_sum) / (2 * h);
        }
    }

    // update linear
    vector<double> update()
    {
        vector<double> update_wb(2);
        weights = weights - learning_rate * gradient_update("weights");
        bias = bias - learning_rate * gradient_update("bias");

        update_wb[0] = weights;
        update_wb[1] = bias;

        return update_wb;
    }
};

class RGBImage {
public:

    // RGB Data ( Vec3b ) -> Vector
    static void convertImageToVector(Mat& image, vector<int>& data) {
        data.clear();
        for (int y = 0; y < image.rows; y++)
        {
            for (int x = 0; x < image.cols; x++)
            {
                Vec3b color = image.at<Vec3b>(Point(x, y));
                data.push_back(color[0]); // B channel
                data.push_back(color[1]); // G channel
                data.push_back(color[2]); // R channel
            }
        }
    }

    // Vector -> Rgb Data( Vec3b )
    static void convertVectorToImage(vector<int>& data, Mat& image) {
        int index = 0;
        for (int y = 0; y < image.rows; y++)
        {
            for (int x = 0; x < image.cols; x++)
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

    // Set Image Data 
    Mat cat_image = imread("cat_image.jpg");
    Mat dog_image = imread("dog_image.jpg");

    // Set Image Size
    Size imageSize = Size(200, 200);
    resize(cat_image, cat_image, imageSize);
    resize(dog_image, dog_image, imageSize);

    vector<int>x_data, y_data;

    // RGB Vector -> Vector
    RGBImage::convertImageToVector(cat_image, x_data);
    RGBImage::convertImageToVector(dog_image, y_data);

    // Set Rand( 0 ~ 1 )
    double weights = ((double)rand() / RAND_MAX);
    double bias = ((double)rand() / RAND_MAX);

    // Set LinearRegression
    LinearRegression lr1;

    lr1.weights = weights;
    lr1.bias = bias;
    lr1.x_data = x_data;
    lr1.y_data = y_data;

    // Train
    for (int i = 0; i < 1500; i++)
    {
        cout << "weights : " << "[" << weights << "]" << " " << "bias : " << "[" << bias << "]";
        cout << "\n";
        vector<double> update_data = lr1.update();
        weights = update_data[0];
        bias = update_data[1];
    }

    cout << "Updated Weights : " << weights << endl;
    cout << "Updated bias : " << bias << endl;

    // validate x_data ( cat image ) to y_data ( dog image )
    vector<int> new_data(x_data.size());
    for (int i = 0; i < x_data.size(); i++) {
        new_data[i] = static_cast<int>(lr1.linear_func(x_data[i], lr1.weights, lr1.bias));
        new_data[i] = min(max(new_data[i], 0), 255);
    }

    // Set predict image
    Mat predict_image = cat_image.clone();
    RGBImage::convertVectorToImage(new_data, predict_image);

    // Show Result
    imwrite("predict_image.jpg", predict_image);
    namedWindow("predict_image", WINDOW_AUTOSIZE);
    imshow("Original Cat image", cat_image);
    imshow("Dog image", dog_image);
    imshow("Predict Cat image", predict_image);
    waitKey(0);

    return 0;
}