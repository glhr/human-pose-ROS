

if (NOT "${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
	message(FATAL_ERROR "Only 64-bit supported on Windows")
endif()

add_compile_definitions(_SCL_SECURE_NO_WARNINGS _CRT_SECURE_NO_WARNINGS NOMINMAX)
add_compile_options(/EHsc) #no asynchronous structured exception handling
add_compile_options(/FS) #no errors beacause of shared pdb writes
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LARGEADDRESSAWARE")

