#include <stdio.h>
#include <iostream>       // std::cout
#include <thread>         // std::thread
#include <cmath>
#include <string>
#include "input_image.cc"
#include "complex.cc"
#include <chrono>

void row(int count, int start, int width, Complex* h, Complex* H){
    
    float twoPi = 6.28318530718;
    int row = start;
    double theta;
    Complex sumVal;
    
    for(int i=0; i< count; i++){ //for each row this thread is responsible for
        
        //this is the for every k loop
        for(int k = 0; k < width; k++){ //index across the row
            
            sumVal = Complex(0,0);
            
            //this is the for every n loop
            for(int n = 0; n < width; n++){
                
                theta = (twoPi*k*n)/width;
                sumVal = sumVal + h[row*width + n] * (Complex(cos(theta), -1*sin(theta)));
                
            }//end n loop
            
            H[row*width + k] = sumVal;
            
        }//end k loop
        
        //row += number of threads;
        //so thread one runs rows 0, 8, 16,24 . . .
        //          two runs rows 1, 9, 17,25 . . .
        //          eight runs    7, 15,23,31 . . .
        row += 8; //change to num threads
        
    }//end count loop
}//end row funct

void col(int count,int start, int width, Complex* h, Complex* H){
    
    float twoPi = 6.28318530718;
    int col = start;
    double theta;
    Complex sumVal;
    
    for(int i=0; i< count; i++){ //for each col this thread is responsible for
        
        //this is the for every k loop
        for(int k = 0; k < width; k++){ //index down the column of H (k is a row number, compute every row)
            
            sumVal = Complex(0,0);
            
            //this is the for every n loop
            for(int n = 0; n < width; n++){ //index down column of h (n is a row number, compute every row)
                
                theta = (twoPi*k*n)/width;
                sumVal = sumVal + h[n*width + col] * (Complex(cos(theta), -1*sin(theta)));
                
            }//end n loop
            
            H[k*width + col] = sumVal;
            
        }//end k loop
        
        //col += number of threads;
        //so thread one runs rows 0, 8, 16,24 . . .
        //          two runs rows 1, 9, 17,25 . . .
        //          eight runs    7, 15,23,31 . . .
        col += 8; //change to num threads
    }
}

int main(int argc, char** argv){
    
    
    //std::string file("T512.txt");
    //if(argc > 1)
    std::string file = std::string(argv[2]);
    
    InputImage image(file.c_str());
    
    Complex * h = image.get_image_data(); //the original time sampled data
    int width = image.get_width();
    int height = image.get_height();
    
    Complex * H = new Complex[width * height]; //the resulting data
    
    
    int count = width/8; //number of rows/columns each thread is responsible for
    
    auto start = std::chrono::high_resolution_clock::now(); //start the clock
    
    //threads that run through the rows
    std::thread one(row, count, 0, width, h, H);
    std::thread two(row, count, 1, width, h, H);
    std::thread three(row, count, 2, width, h, H);
    std::thread four(row, count, 3, width, h, H);
    std::thread five(row, count, 4, width, h, H);
    std::thread six(row, count, 5, width, h, H);
    std::thread seven(row, count, 6, width, h, H);
    std::thread eight(row, count, 7, width, h, H);
    
    //wait for all threads to finish before continuing on to the columns
    one.join();
    two.join();
    three.join();
    four.join();
    five.join();
    six.join();
    seven.join();
    eight.join();
    
    //compute on all of the columns
    std::thread onen(col, count, 0, height, H, h);
    std::thread twon(col, count, 1, height, H, h);
    std::thread threen(col, count, 2, height, H, h);
    std::thread fourn(col, count, 3, height, H, h);
    std::thread fiven(col, count, 4, height, H, h);
    std::thread sixn(col, count, 5, height, H, h);
    std::thread sevenn(col, count, 6, height, H, h);
    std::thread eightn(col, count, 7, height, H, h);
    
    //wait for everything to finish
    onen.join();
    twon.join();
    threen.join();
    fourn.join();
    fiven.join();
    sixn.join();
    sevenn.join();
    eightn.join();
    
    //stop the clock and compute execution time
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    std::cout << "Time taken by CPU threads: "
    << duration.count() << " microseconds" << std::endl;
    
    //std::string outFile("512out.txt");
    std::string outFile = std::string(argv[3]);
    
    //write output file, h is the results after the columns have been computed
    //reusing that allocated memory block saves memory usage
    image.save_image_data(outFile.c_str(), h, width, height);
    
    return 0;
    
}
