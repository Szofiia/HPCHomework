#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>


using namespace cv;
using namespace std;

// Read image file from arguments
void processImageFromFile(Mat& I, char* fileName)
{
    // Read
    Mat image = imread(fileName, IMREAD_GRAYSCALE);

    if (image.empty()) // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        exit(0);
    }

    // Padding for faster Fourier Transform
    Mat Ip;
    int M = getOptimalDFTSize(image.rows);
    int N = getOptimalDFTSize(image.cols);

    if (M != N) // Check for invalid input
    {
        cout << "Image width and height do not match" << std::endl;
        exit(0);
    }

    copyMakeBorder(image, Ip, 0, M - image.rows, 0, N - image.cols, BORDER_CONSTANT, Scalar::all(0));

    // Copy to I
    Ip.copyTo(I);
}

// Create nxn Watermark image 
void createWatermark(Mat& W, int n)
{
    RNG rng;
    W = Mat(n, n, CV_32F);
    cv::randu(W, 0, 1);
}

int main(int argc, char** argv)
{
    std::cout << "SS Watermarking\n";

    if (argc != 2)
    {
        cout << "Error: Image not added as an argument." << endl;
        return -1;
    }

    // Load image
    Mat I;
    processImageFromFile(I, argv[1]);
    imshow("Original Image", I);

    // Create Watermark
    int n = 8;
    Mat W;
    createWatermark(W, n);

    // Write into file, png conversion
    Mat Iw;
    W.convertTo(Iw, CV_8U, 255.0);
    imwrite("watermark.png", Iw);

    imshow("Watermark", Iw);

    waitKey(0);
    return 0;

}
