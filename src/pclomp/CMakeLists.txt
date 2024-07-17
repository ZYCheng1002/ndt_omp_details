add_library(ndt_omp
        voxel_grid_covariance_omp.cpp
        ndt_omp.cpp
)

add_dependencies(ndt_omp
        ${PROJECT_NAME}.tools
)

add_library(gicp_omp
        gicp_omp.cpp
)

add_dependencies(gicp_omp
        ${PROJECT_NAME}.tools
)