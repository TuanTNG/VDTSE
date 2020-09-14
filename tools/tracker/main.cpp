#include "array.h"
#include "tracker.h"
#include <cstdint>
#include <iostream>

extern "C" {
int track(float t, void* det_ptr, int len, void* ret_ptr, void* count_ptr){
	array<std::int32_t> det(det_ptr, len, 5);
	array<std::int32_t> ret(ret_ptr, 256, 7);
	array<std::int32_t> counter(count_ptr, 256, 1);

	track(t, det);
	return write(ret, counter);
}

void camera_info(int w, int h, float fov, float alpha, float H){
    cam_info::w = w;
    cam_info::h = h;
    cam_info::fov = fov;
    cam_info::alpha = alpha;
    cam_info::H = H;
    std::cout << w << ' ' << h << ' ' << fov << ' ' << alpha << ' ' << H << '\n';
}

}
