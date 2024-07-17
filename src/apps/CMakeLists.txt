add_executable(align
        align.cpp
)
# 作用: align依赖ndt_omp, 会先编译ndt_omp
add_dependencies(align
        ndt_omp
        gicp_omp
)

target_link_libraries(align
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ndt_omp
        gicp_omp
        ${PROJECT_NAME}.tools
        glog
)

add_executable(ndt_omp_test
        ndt_omp_test.cpp
)
# 作用: align依赖ndt_omp, 会先编译ndt_omp
add_dependencies(ndt_omp_test
        ndt_omp
        gicp_omp
)

target_link_libraries(ndt_omp_test
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ndt_omp
        gicp_omp
        ${PROJECT_NAME}.tools
        glog
)