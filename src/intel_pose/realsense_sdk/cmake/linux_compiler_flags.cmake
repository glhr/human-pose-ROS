# warning flags
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Werror=return-type ")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized -Winit-self" )
if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmaybe-uninitialized")
endif()

# pthread flags in order to allow using it (used by intel::inferenceengine library)
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)


#Rpath handling
SET(CMAKE_INSTALL_RPATH "$ORIGIN/../lib;$ORIGIN/../lib/dep;$ORIGIN/dep;$ORIGIN")

# intrinsics flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*")
        message(STATUS "Setting compiler options for GNUCXX compiler")
        add_definitions(-DSSE_INTRINSICS_AVAILABLE)
        add_compile_options("-ffast-math")
        add_compile_options("-fno-omit-frame-pointer")
        add_compile_options("-msse")
        add_compile_options("-msse2")
        add_compile_options("-msse3")
        add_compile_options("-mssse3")
        add_compile_options("-msse4.1")
        add_compile_options("-msse4.2")
endif()

# specify that the libraries will be dynamically linked
set (LIB_DL dl)
