project(Diezel)

cmake_minimum_required(VERSION 3.8.2)

# ======================== CUDA ========================
find_package(CUDA)

if(CUDA_FOUND)

    enable_language(CUDA)

    try_run(
    RUN_RESULT_VAR 
    COMPILE_RESULT_VAR
    ${CMAKE_CURRENT_BINARY_DIR}/ 
    ${PROJECT_SOURCE_DIR}/util/CheckIfCudaDeviceIsPresent.cu
    COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT_VAR
    RUN_OUTPUT_VARIABLE RUN_OUTPUT_VAR)
    if("${COMPILE_RESULT_VAR}" AND ("${RUN_RESULT_VAR}" GREATER 0))
        message("-- Found : ${RUN_RESULT_VAR} cuda capable device(s)")    
        set(CUDA_HAVE_GPU TRUE)
    else()
        message("-- Found cuda packages but there's no cuda capable device")
        set(CUDA_HAVE_GPU FALSE)
    endif()

else()
    message("-- Cuda packages not found")
endif(CUDA_FOUND)

# ======================== TEST ========================
enable_testing()
if(CUDA_FOUND AND CUDA_HAVE_GPU)

    #======================== VECTOR =====================
    add_executable(vectorUnitTest ${PROJECT_SOURCE_DIR}/test/VectorUnitTest.cpp)

    target_include_directories(
        vectorUnitTest
        PUBLIC  ${PROJECT_SOURCE_DIR}/include/
        PRIVATE ${PROJECT_SOURCE_DIR}/src/
    )

    target_sources(
        vectorUnitTest
        PUBLIC  ${PROJECT_SOURCE_DIR}/include/LinearAlgebra.hpp
        PRIVATE ${PROJECT_SOURCE_DIR}/src/Vector.cu
        PRIVATE ${PROJECT_SOURCE_DIR}/src/Matrix.cu
        PRIVATE ${PROJECT_SOURCE_DIR}/src/CSRMatrix.cu
    )

    enable_testing()

    add_test(
        NAME vectorUnitTest
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build/
        COMMAND ./vectorUnitTest
    )

    #======================== MATRIX ======================
    add_executable(matrixUnitTest ${PROJECT_SOURCE_DIR}/test/MatrixUnitTest.cpp)

    target_include_directories(
        matrixUnitTest
        PUBLIC  ${PROJECT_SOURCE_DIR}/include/
        PRIVATE ${PROJECT_SOURCE_DIR}/src/
    )

    target_sources(
        matrixUnitTest
        PUBLIC  ${PROJECT_SOURCE_DIR}/include/LinearAlgebra.hpp
        PRIVATE ${PROJECT_SOURCE_DIR}/src/Vector.cu
        PRIVATE ${PROJECT_SOURCE_DIR}/src/Matrix.cu
        PRIVATE ${PROJECT_SOURCE_DIR}/src/CSRMatrix.cu
    )

    enable_testing()

    add_test(
        NAME matrixUnitTest
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build/
        COMMAND ./matrixUnitTest
    )
    #======================== CSRMATRIX ===================
    add_executable(csrMatrixUnitTest ${PROJECT_SOURCE_DIR}/test/CSRMatrixUnitTest.cpp)

    target_include_directories(
        csrMatrixUnitTest
        PUBLIC  ${PROJECT_SOURCE_DIR}/include/
        PRIVATE ${PROJECT_SOURCE_DIR}/src/
    )

    target_sources(
        csrMatrixUnitTest
        PUBLIC  ${PROJECT_SOURCE_DIR}/include/LinearAlgebra.hpp
        PRIVATE ${PROJECT_SOURCE_DIR}/src/Vector.cu
        PRIVATE ${PROJECT_SOURCE_DIR}/src/Matrix.cu
        PRIVATE ${PROJECT_SOURCE_DIR}/src/CSRMatrix.cu
    )

    enable_testing()

    add_test(
        NAME csrMatrixUnitTest
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build/
        COMMAND ./csrMatrixUnitTest
    )

    
endif()