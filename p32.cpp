//
//  p32.cpp
//  Creators: 
//
//  Vishal Devidas
//  Dave Patel
//
//  


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include "input_image.h"
#include "complex.h"
#include <fstream>
//#include <chrono>

using namespace std;

#define NUM_RANK    8

const float PI = 3.14159265358979f;

// performs necessary bitreversal for performance improvements
void bitReverse(Complex *vec, int size){
    
    int i = 0;
    int j;
      
    for(int n = 0; n < size-2; n++){
        if(n < i){

            Complex temp = vec[i];
            vec[i] = vec[n];
            vec[n] = temp;

        }
         
        j = size/2;
        while(j <= i){

            i -= j;
            j /= 2;
        
        } 
        i += j;
   }
   
}

// Cooley - Tukey FFT Algorithm
void fft_CT(Complex* array, Complex* w, int size, int power){
    float rel,img;
    Complex product;
    
    bitReverse(array, size);
    
    for(int n = 2; n < size; n *= 2){
        power--;
        
        for(int i = 0; i < n/2; i++){
            for(int j = 0; j < size; j += n) {
                Complex a = array[i + j + (n/2)];
                Complex b = w[i<<power];
                
                product = a * b;
                
                rel = product.real;
                img = product.imag;
                
                array[i + j + (n/2)].real = array[i + j].real - rel;
                array[i + j + (n/2)].imag = array[i + j].imag - img;
                
                array[i + j].real = array[i + j].real + rel;
                array[i + j].imag = array[i + j].imag + img;
                
            }
        }
    }
    
    for(int i = 0; i < (size/2); i++){
        Complex a = array[i + (size/2)];
        Complex b = w[i];
        
        product = a * b;
        
        rel = product.real;
        img = product.imag;
        
        array[i + (size/2)].real = array[i].real - rel;
        array[i + (size/2)].imag = array[i].imag - img;
        
        array[i].real = array[i].real + rel;
        array[i].imag = array[i].imag + img;
    }
    
}

int main(int argc, char *argv[]){
    
    //auto start = std::chrono::system_clock::now();
    //clock_t begin = clock();
    std::string input;
    std::string output;
    //InputImage img("Tower256.txt");
    if(argc > 2){
        input = std::string(argv[2]);
        output = std::string(argv[3]);
    }else{
        cout << "need to include inputfile and outputfile";
        return 0;
    }

    InputImage img(input.c_str());
    
    int num_rows = img.get_height();
    int num_col = img.get_width();
    int n = num_rows;
    int gridSize = num_col * num_rows;
    int numRanks;
    int rank;
    int size = num_col;
    int Section = (size / NUM_RANK);
    int power = log2(num_col);
    
    
    Complex * pic1D = img.get_image_data();
    Complex picFinal1D[num_col*num_rows];
    Complex pic[size][size];
    for(int i = 0;i<size;i++) {
        for(int j = 0;j<size;j++) {
            pic[i][j]=pic1D[(i*size)+j];
        }
    }
    
    Complex picSection[Section][size];
    Complex picWhole[NUM_RANK][Section][Section];
    Complex pic2[size][size];
    Complex pic2Section[size][Section];
    Complex* finalImg;
    Complex w[num_col/2];
    
    clock_t begin = clock();

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    for(int i = 0; i < size/2; i++){
        
        w[i].real = cos(((2*PI)*i)/size);
        w[i].imag = (-1*sin(((2*PI)*i)/size));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    MPI_Scatter((char *) pic, Section * size * 2, MPI_FLOAT, (char*) picSection, Section * size * 2, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    for(int i = 0; i < Section; i++){
        fft_CT(&picSection[i][0], w, size, power);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    for(int procID = 0; procID < NUM_RANK; procID++){
        for(int i = 0; i < Section; i++){
            for(int j = 0; j < Section; j++){
                picWhole[procID][i][j].real = picSection[i][j+(Section*procID)].real;
                picWhole[procID][i][j].imag = picSection[i][j+(Section*procID)].imag;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Alltoall(picWhole, Section * Section * 2, MPI_FLOAT, pic2Section, Section * Section * 2, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    for(int i = 0; i < Section; i++){
        for(int j = 0; j < num_col; j++){
            picSection[i][j].real = pic2Section[j][i].real;
            picSection[i][j].imag = pic2Section[j][i].imag;
        }
    }
    
    for(int i = 0; i < Section; i++){
        fft_CT(&picSection[i][0], w, num_col, power);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    finalImg = (Complex*)malloc(size * size * sizeof(Complex));
    
    MPI_Gather(picSection, Section * size * 2, MPI_FLOAT, pic, Section * size * 2, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
   
    if(rank == 0){
        ofstream OutputFile(output.c_str());
        //for(int i = 0; i< num_col; i++) {
        //    OutputFile<<"("<<pic[i].real<<", "<<pic[i].imag<<")\n";
        //}
        OutputFile.close();

	    clock_t end = clock();
        double time_spent = ((double)(end-begin)) / CLOCKS_PER_SEC;
        cout << "time is " << time_spent << " s" << endl;
        //ofstream timeFile("time1024.txt");
        //timeFile << "Time = " << time_spent << " s";
        //timeFile.close();
        
        for(int i = 0; i < size; i++){
            for(int j = 0; j < size; j++){
                pic2[i][j].real = pic[j][i].real;
                pic2[i][j].imag = pic[j][i].imag;
            }
        }
        for(int i = 0; i < size; i++){
            for(int j = 0; j < size; j++){
                picFinal1D[(i*size)+j].real = pic2[i][j].real;
                picFinal1D[(i*size)+j].imag = pic2[i][j].imag;
            }
        }
        
        /*for(int i = 0;i<size;i++) {
            for(int j = 0;j<size;j++) {
                pic1D[(i*size)+j]=pic2[i][j];
            }
        }*/
        
	    InputImage example(output.c_str());
        example.save_image_data(output.c_str(), picFinal1D, num_col, num_rows);
        
	
    }
       /* if (rank == 0) {
       //   auto end = std::chrono::system_clock::now();
       //   std::chrono::duration<double> elapsed_seconds = end-start;
       //   std::cout << elapsed_seconds.count() << std::endl;
            clock_t end = clock();
            double time_spent = (double)((end-begin) / CLOCKS_PER_SEC);
            cout << "time is " << time_spent << " s" << endl;
            ofstream timeFile("time1024.txt");
            timeFile << "Time = " << time_spent << " s";
            timeFile.close
          }
       */
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    //delete [] pic;
    return 0;
}
