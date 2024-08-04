#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class LinearRegression {
public:
    double weights;
    double bias;
    double learning_rate = 0.1; // 학습률 조정
    vector<double> x_data;
    vector<double> y_data;

public:
    // wx + b
    double linear_func(double _x_data, double _weights, double _bias) {
        return _x_data * _weights + _bias;
    }

    // MSE loss function
    double loss_func(double _weights, double _bias)
    {
        double loss = 0.0;
        for (int idx = 0; idx < x_data.size(); ++idx)
        {
            double diff = linear_func(x_data[idx], _weights, _bias) - y_data[idx];
            loss += diff * diff;
        }
        return loss / x_data.size();
    }

    // update gradient ( central difference )
    double gradient_update(string grad_name)
    {
        double h = 1e-5; // 적절한 작은 값 설정
        if (grad_name == "weights")
        {
            double forward = loss_func(weights + h, bias);
            double backward = loss_func(weights - h, bias);
            return (forward - backward) / (2 * h);
        }
        else {
            double forward = loss_func(weights, bias + h);
            double backward = loss_func(weights, bias - h);
            return (forward - backward) / (2 * h);
        }
    }

    // update linear
    vector<double> update()
    {
        vector<double> update_wb(2);
        double grad_w = gradient_update("weights");
        double grad_b = gradient_update("bias");
        weights -= learning_rate * grad_w;
        bias -= learning_rate * grad_b;

        update_wb[0] = weights;
        update_wb[1] = bias;

        return update_wb;
    }
};

class RGBImage {
public:
    // RGB Data ( Vec3b ) -> Vector
    static void convertImageToVector(Mat& image, vector<double>& data) {
        data.clear();
        for (int y = 0; y < image.rows; y++)
        {
            for (int x = 0; x < image.cols; x++)
            {
                Vec3b color = image.at<Vec3b>(Point(x, y));
                data.push_back(color[0] / 255.0); // B channel
                data.push_back(color[1] / 255.0); // G channel
                data.push_back(color[2] / 255.0); // R channel
            }
        }
    }

    // Vector -> RGB Data ( Vec3b )
    static void convertVectorToImage(vector<double>& data, Mat& image) {
        int index = 0;
        for (int y = 0; y < image.rows; y++)
        {
            for (int x = 0; x < image.cols; x++)
            {
                Vec3b& color = image.at<Vec3b>(Point(x, y));
                color[0] = static_cast<int>(min(max(data[index++], 0.0), 1.0) * 255); // B channel
                color[1] = static_cast<int>(min(max(data[index++], 0.0), 1.0) * 255); // G channel
                color[2] = static_cast<int>(min(max(data[index++], 0.0), 1.0) * 255); // R channel
            }
        }
    }
};

// Global variables for trackbar
vector<Mat> predicted_images;
int slider_pos = 0;

void on_trackbar(int, void*) {
    imshow("Predict Cat image", predicted_images[slider_pos]);
}

int main()
{
    // Set Image Data 
    Mat cat_image = imread("C:/Users/82109/Desktop/visionStudy/cat_image.PNG");
    Mat dog_image = imread("C:/Users/82109/Desktop/visionStudy/dog_image.PNG");

    if (cat_image.empty() || dog_image.empty()) {
        cout << "Could not open or find the images!" << endl;
        return -1;
    }

    // Set Image Size
    Size imageSize = Size(200, 200);
    resize(cat_image, cat_image, imageSize);
    resize(dog_image, dog_image, imageSize);

    vector<double> x_data, y_data;

    // RGB Vector -> Vector
    RGBImage::convertImageToVector(cat_image, x_data);
    RGBImage::convertImageToVector(dog_image, y_data);

    // Print initial data
    cout << "Initial x_data (first 10 elements): ";
    for (int i = 0; i < 10; ++i) {
        cout << x_data[i] << " ";
    }
    cout << endl;

    cout << "Initial y_data (first 10 elements): ";
    for (int i = 0; i < 10; ++i) {
        cout << y_data[i] << " ";
    }
    cout << endl;

    // Set Rand( 0 ~ 1 )
    double weights = ((double)rand() / RAND_MAX);
    double bias = ((double)rand() / RAND_MAX);

    // Print initial weights and bias
    cout << "Initial weights: " << weights << ", Initial bias: " << bias << endl;

    // Set LinearRegression
    LinearRegression lr1;

    lr1.weights = weights;
    lr1.bias = bias;
    lr1.x_data = x_data;
    lr1.y_data = y_data;

    // Train and save intermediate results
    for (int i = 0; i < 2000; i++)
    {
        vector<double> update_data = lr1.update();
        weights = update_data[0];
        bias = update_data[1];

        if (i % 100 == 0) {
            vector<double> new_data(x_data.size());
            for (int j = 0; j < x_data.size(); j++) {
                new_data[j] = lr1.linear_func(x_data[j], lr1.weights, lr1.bias);
                new_data[j] = min(max(new_data[j], 0.0), 1.0);
            }
            Mat predict_image = cat_image.clone();
            RGBImage::convertVectorToImage(new_data, predict_image);
            predicted_images.push_back(predict_image);

            cout << "Iteration " << i << ": weights = " << weights << ", bias = " << bias << ", loss = " << lr1.loss_func(weights, bias) << endl;
        }
    }

    cout << "Updated Weights : " << weights << endl;
    cout << "Updated bias : " << bias << endl;

    namedWindow("Predict Cat image", WINDOW_AUTOSIZE);
    createTrackbar("Iterations", "Predict Cat image", &slider_pos, predicted_images.size() - 1, on_trackbar);
    on_trackbar(slider_pos, 0);

    imshow("Original Cat image", cat_image);
    imshow("Dog image", dog_image);
    waitKey(0);

    return 0;
}
