#include "array.h"
#include <set>
#include "tracker.h"
#include "iou.h"
#include <cmath>

#include <iostream>


namespace cam_info{
    int w, h;
    float alpha, fov, H;
}

struct Object{
	const int id;
	mutable int cls;

	mutable bool counted;
	mutable bool visible; //even when object disappears, we still keep tracking for sometimes

	mutable int x, y, w, h;
	mutable float t;
	mutable float v;

	mutable bool fully_appeared; // false if it has just appeared at the edge of the image
	mutable float ot;
	mutable int ox, oy, ow, oh;

	Object(float t, int id, int cls, int x, int y, int w, int h);

	void update(float t, int cls, int x, int y, int w, int h) const;

	float iou(const Object&) const;
	float iou(int x, int y, int w, int h) const;
	bool at_edge() const;

	friend bool operator<(const Object& a, const Object& b){
		return a.id < b.id;
	}
};

Object::Object(float t, int id, int cls, int x, int y, int w, int h):
t(t),
id(id),
cls(cls),
x(x), y(y), w(w), h(h)
{
    v = -1;
	counted = false;
	fully_appeared = false;
    if (!at_edge()){
        fully_appeared = true;
        ot = t;
        ox = x;
        oy = y;
        oh = h;
        ow = w;
    }
    visible = true;
}

void Object::update(float t, int cls, int x, int y, int w, int h) const{
	this -> t = t;
	this -> cls = cls;
	this -> x = x;
	this -> y = y;
	this -> w = w;
	this -> h = h;
	if(!fully_appeared && !at_edge()){
        fully_appeared = true;
        ot = t;
        ox = x;
        oy = y;
        ow = w;
        oh = h;
    }
}
// cho tuan
float Object::iou(const Object& other) const{
	int x1 = this -> x;
	int y1 = this -> y;
	int w1 = this -> w;
	int h1 = this -> h;

	int x2 = other.x;
	int y2 = other.y;
	int w2 = other.w;
	int h2 = other.h;

	return ::iou(x1, y1, w1, h1, x2, y2, w2, h2);
}

float Object::iou(int x, int y, int w, int h) const{
	int x1 = this -> x;
	int y1 = this -> y;
	int w1 = this -> w;
	int h1 = this -> h;
	return ::iou(x1, y1, w1, h1, x, y, w, h);
}

bool Object::at_edge() const{
    int w01 = cam_info::w / 20;
    int h01 = cam_info::h / 20;
    if (x - w / 2 < w01)
        return true;
    if (x + w / 2 > cam_info::w - w01)
        return true;
    if (y + h / 2 > cam_info::h - h01)
        return true;
	if (y - h / 2 < h01)
        return true;
    return false;
}


// tracker variables
static std::set<Object> data;
static bool id[256];
static int counter[256];

static int get_id(){
	for (int i = 0; i < sizeof(id)/sizeof(bool); ++i)
		if (!id[i]){
			id[i] = true;
			return i;
		}
    #ifdef DEBUG
    std::cerr << "Number of vehicle exceed the limit (256) \n";
    #endif // DEBUG
	return -1;
}

static void update(float t, int cls, int x, int y, int w, int h){
    // looking for tracked objects that best match
	float best_iou = 0.0;
	const Object *p;
	for (auto& obj: data){
		if (cls != obj.cls) continue;
		float iou = obj.iou(x, y, w, h);
		if (iou > best_iou){
			best_iou = iou;
			p = &obj;
		}
	}

    // if there is a match, update
	if (best_iou > 0.5)
		p -> update(t, cls, x, y, w, h);

    // if there is no match, create new
	else {
		Object new_obj(t, get_id(), cls, x, y, w, h);
		data.insert(new_obj);
	}
}

static float distance(float r1, float p1, float r2, float p2){
	return std::sqrt(r1 * r1 + r2 * r2 - 2 * r1 * r2 * std::cos(p2 - p1));
}

static void calc_velocity(){
	for (auto& obj: data){
		if (obj.t - obj.ot < 0.1 || obj.fully_appeared == false){
			obj.v = -1;
			continue;
		}

		if (obj.at_edge()) continue;

		float x, y, x0, y0;
		x0 = obj.ox;
		y0 = obj.oy + obj.oh / 2; // we use bottom edge for calculating velocity
		x = obj.x;
		y = obj.y + obj.h / 2;

		float w = float(cam_info::w);
		float h = float(cam_info::h);
		float& fov = cam_info::fov;
		float fovv = fov * h / w; //field of view vertical
		float& alpha = cam_info::alpha;
		float& H = cam_info::H;

		float r0 = H * std::tan(alpha + fovv * (0.5 - y0 / h));
		float p0 = 1 / std::sin(alpha + fovv * (0.5 - y0 / h)) * fov * x0 / w;

		float r = H * std::tan(alpha + fovv * (0.5 - y / h));
		float p = 1 / std::sin(alpha + fovv * (0.5 - y / h)) * fov * x / w;

		float d = distance(r0, p0, r, p);

		obj.v = d / (obj.t - obj.ot) * 3.6;
	}
}

void track(float t, array<std::int32_t>& x){
	for (int i = 0; i < x.shape(0); ++i)
		update(t, x.at(i, 4), x.at(i, 0), x.at(i, 1), x.at(i, 2), x.at(i, 3));

	for (auto p = data.begin(); p != data.end();)
		if (t - p -> t > 0.25){
            id[p -> id] = false;
			p = data.erase(p);
		}
		else ++p;

	calc_velocity();

	for (auto& obj: data)
        obj.visible = (t == obj.t);
	
	for (auto& obj: data)
		if (!obj.counted && obj.t - obj.ot > 0.5){
			obj.counted = true;
			counter[obj.cls]++;
		}
}

int write(array<std::int32_t>& track_data, array<std::int32_t>& count_data){
	for (int n = 0; n < 256; ++n)
		count_data.at(n, 0) = counter[n];

	int i = 0;
	for (auto obj: data){
        if (!obj.visible)
            continue;
		track_data.at(i, 0) = obj.cls;
		track_data.at(i, 1) = obj.x;
		track_data.at(i, 2) = obj.y;
		track_data.at(i, 3) = obj.w;
		track_data.at(i, 4) = obj.h;
		track_data.at(i, 5) = obj.v;
		track_data.at(i, 6) = obj.id;
		++i;
	}
	return i;

}
