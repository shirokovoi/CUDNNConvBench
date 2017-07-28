if (NOT DEFINED CUDNN_PATH)
	set(CUDNN_PATH /usr/local/cudnn)
endif()

message(STATUS "Searching for CUDNN in ${CUDNN_PATH}")

find_path(CUDNN_INCLUDE_PATH cudnn.h PATHS ${CUDNN_PATH}/include)
if (NOT CUDNN_INCLUDE_PATH)
	message(FATAL_ERROR "Can't find cudnn.h")
endif()

find_library(CUDNN_LIB cudnn PATH_SUFFIXES lib64 lib)
if (NOT CUDNN_LIB)
	message(FATAL_ERROR "Can't find libcudnn")
endif()

set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_PATH})
set(CUDNN_LIBRARIES ${CUDNN_LIB})
set(CUDNN_FOUND TRUE)

if (NOT TARGET CUDNN)
	add_library(CUDNN SHARED IMPORTED)
	set_target_properties(CUDNN PROPERTIES IMPORTED_LOCATION "${CUDNN_LIB}")
	set_target_properties(CUDNN PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIRS}")
endif()