#ifndef __ARRAY__H_
#define __ARRAY__H_

#ifdef DEBUG
#include <iostream>
#endif

template<typename dtype>
class array{
private:
    dtype* data;
    int num_elem;
    int num_comp;
public:
    array(void* data, int n, int c){
        this -> data = (dtype*)data;
        num_elem = n;
        num_comp = c;
    }

    dtype at(int n, int c) const{
        #ifdef DEBUG
        if (n >= num_elem || c >= num_comp)
            std::cerr << "Out of range:(" << n << ", " << c << ") in (" << num_elem << ", " << num_comp << '\n';
        #endif
        return *(data + num_comp * n + c);
    }
    dtype& at(int n, int c){
        #ifdef DEBUG
        if (n >= num_elem || c >= num_comp)
            std::cerr << "Out of range:(" << n << ", " << c << ") in (" << num_elem << ", " << num_comp << '\n';
        #endif
        return *(data + num_comp * n + c);
    }

    int shape(int n) const{
        #ifdef DEBUG
        if (n > 1)
            std::cerr << "Shape (" << n << ") can't be greater than 1\n";
        #endif

        if (n == 0)
            return num_elem;
        else
            return num_comp;
    }
};

#endif
