project(tracker)
add_library(track SHARED main.cpp tracker.cpp)
set_target_properties(track PROPERTIES SUFFIX "" PREFIX "")
