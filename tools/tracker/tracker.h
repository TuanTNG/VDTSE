#ifndef __TRACKER_H__
#define __TRACKER_H__
#include "array.h"
#include <cstdint>
namespace cam_info{
extern int w, h;
extern float alpha, fov, H;
}
void track(float t, array<std::int32_t>& x);
int write(array<std::int32_t>& x, array<std::int32_t>& count);


#endif
