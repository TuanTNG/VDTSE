template<typename T>
static inline T max(T a, T b){
    return (a > b)? a : b;
}

template<typename T>
static inline T min(T a, T b){
    return (a > b)? b : a;
}

template<typename T>
static inline float iou(T x1, T y1, T w1, T h1, T x2, T y2, T w2, T h2){
    T x_left = max(x1 - w1 / 2, x2 - w2 / 2);
    T x_right = min(x1 + w1 / 2, x2 + w2 / 2);
    T y_top = max(y1 - h1 / 2, y2 - h2 / 2);
    T y_bot = min(y1 + h1 / 2, y2 + h2 / 2);
    if (x_right < x_left || y_bot < y_top)
        return 0;
    T area1 = w1 * h1;
    T area2 = w2 * h2;
    T intersec = (x_right - x_left) * (y_bot - y_top);
    
    return float(intersec) / float((area1 + area2 - intersec));
}

template<typename T>
static inline float iou(T* box1, T* box2){
	return iou(box1[0], box1[1], box1[2], box1[3], box2[0], box2[1], box2[2], box2[3]);
}