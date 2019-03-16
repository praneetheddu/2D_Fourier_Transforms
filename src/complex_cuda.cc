//
// Created by brian on 11/20/18.
//

#include "complex_cuda.h"

#include <cmath>

const float PI = 3.14159265358979f;

__host__ __device__ Complex::Complex() : real(0.0f), imag(0.0f) {}

__host__ __device__ Complex::Complex(float r) : real(r), imag(0.0f) {}

__host__ __device__ Complex::Complex(float r, float i) : real(r), imag(i) {}

__host__ __device__ Complex Complex::operator+(const Complex &b) const {
    return Complex(b.real + real, b.imag + imag);
}

__host__ __device__ Complex Complex::operator-(const Complex &b) const {
    return Complex(real - b.real, imag - b.imag);
}

__host__ __device__ Complex Complex::operator*(const Complex &b) const {
    return Complex(b.real*real - b.imag*imag, real*b.imag + imag*b.real);
}

__host__ __device__ Complex Complex::mag() const {
    return Complex(sqrt(real*real + imag*imag));
}

__host__ __device__ Complex Complex::angle() const { //angle in radians, easier/quicker for transform calcs
    return Complex(atan(imag/real));
}

__host__ __device__ Complex Complex::conj() const {
    return Complex(real, -imag);
}

std::ostream& operator<< (std::ostream& os, const Complex& rhs) {
    Complex c(rhs);
    if(fabsf(rhs.imag) < 1e-10) c.imag = 0.0f;
    if(fabsf(rhs.real) < 1e-10) c.real = 0.0f;

    if(c.imag == 0) {
        os << c.real;
    }
    else {
        os << "(" << c.real << "," << c.imag << ")";
    }
    return os;
}
