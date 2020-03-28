
find_package( OpenCV REQUIRED )


add_subdirectory(evaluate)

include_directories(${OpenCV_INCLUDE_DIRS} )
include_directories(evaluate)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

add_library(feature SHARED "feature.cpp")
add_library(bucket SHARED "bucket.cpp")
add_library(utils SHARED "utils.cpp")
add_library(visualOdometry SHARED "visualOdometry.cpp")
add_library(rgbd_standalone SHARED "rgbd_standalone.cpp")

add_library(Frame SHARED "Frame.cpp")

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -lineinfo
    # -gencode=arch=compute_30,code=sm_30
    )
#add_library(cuda_test "cuda_test.cuh")
#target_compile_features(cuda_test PUBLIC cxx_std_11)
#set_target_properties( cuda_test
#                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
message(${PROJECT_SOURCE_DIR})

cuda_add_executable( cuda_test cuda_test.cu )
cuda_add_executable( run main.cpp solve_pnp_custom.cu )
target_link_libraries( bucket ${OpenCV_LIBS} )
target_link_libraries( feature ${OpenCV_LIBS} bucket)
target_link_libraries( utils evaluate_odometry feature ${OpenCV_LIBS} )
target_link_libraries( visualOdometry utils bucket feature ${OpenCV_LIBS})


target_link_libraries( Frame ${OpenCV_LIBS} )
target_link_libraries( run ${OpenCV_LIBS} bucket feature utils visualOdometry Frame rgbd_standalone )
