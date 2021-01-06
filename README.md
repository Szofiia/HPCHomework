# HPCHomework
This repository is responsible for containing:

* the Midterm Homework

of High Performance Parallel Copmuting class of BME VIK, 2020 fall semester.

## Build on Windows (x64)

### OpenCV
1. Download [OpenCV](https://opencv.org/releases/) for Windows.
1. Install opencv-*.exe in

```sh
$  C:\YourDirectory\
```

3. Go into system environment variables and scroll down to the "Path" variable, edit it and add the following directory (system restart is usually needed):

```
C:\YourDirectory\opencv\build\x64\vc15\bin
```

4. Open Visual Studio, and open to Project properties
4. Change *Configuration* to **Debug** and *Platform* to **x64**
4. In *Configuration Properties/VC++ Directories*, edit *Include Directories* to match the following:

```
C:\YourDirectory\opencv\build\include;$(IncludePath)
```

6. In *Configuration Properties/VC++ Directories*, edit *Library Directories* to match the following:

```
C:\YourDirectory\opencv\build\x64\vc15\lib;C:\YourDirectory\opencv\build\x64\vc15\bin;$(LibraryPath)
```

7. In *Linker/Input*, edit *Additional Dependencies* to match the following:

```
opencv_world*d.lib;%(AdditionalDependencies)
```

8. Apply changes
8. Try the following demo [source code](https://github.com/opencv/opencv/blob/master/samples/cpp/tutorial_code/introduction/windows_visual_studio_opencv/introduction_windows_vs.cpp) and put [this image](https://github.com/opencv/opencv/blob/master/samples/data/opencv-logo.png) in the project folder:

```c++
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace cv;
using namespace std;
int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<" Usage: " << argv[0] << " ImageToLoadAndDisplay" << endl;
     return -1;
    }
    Mat image;
    image = imread(argv[1], IMREAD_COLOR); // Read the file
    if( image.empty() ) // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl ;
        return -1;
    }
    namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    imshow( "Display window", image ); // Show our image inside it.
    waitKey(0); // Wait for a keystroke in the window
    return 0;
}
```

10. Build and Run the code by setting Command Arguments

## Concept

Digital watermarting is a partial solution for a problem of illegally obtaining copyrighted material.
Several digital watermarking methods exist in the spatioal and frequency domain also. Digital image
watermarking can be achieved by applying the discrete cosine transform (DCT) on the image and
embedding the watermark in the „frequency domain”, then using the inverse transformation to recompose the image.

The goal is to implement the sequential algorithm of some kind of digital
watermarking technique in the frequency domain (possibly based on DFT or DTFT). Then, to implement
the paralellised version of the algorithm to use available resources of the GPU and raise effectiveness.
By comparing the two versions, we can conclude how paralellisation improves performance

## High Performance Parallel Computing at BME
* [Course Website](https://www.iit.bme.hu/targyak/BMEVIIIMA06)
* Professor: Dr. Imre Szeberényi