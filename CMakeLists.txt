set(BUNGEE_VERSION 0.0.0 CACHE STRING "Bungee version string")
set(BUNGEE_SELF_TEST 0 CACHE STRING "Enable Bungee self tests (0=off, 1=fast, 2=full)")
option(BUNGEE_BUILD_SHARED_LIBRARY "Build shared, dynamic Bungee library in addition to a static libary" ON)

cmake_minimum_required(VERSION 3.30...3.31)

project(bungee VERSION ${BUNGEE_VERSION})
set(CMAKE_CXX_STANDARD 20)

if (WIN32)
  set(BUNGEE_EXPORT "__declspec(dllexport)")
else()
  set(BUNGEE_EXPORT "__attribute__((visibility(\"default\")))")
endif()

# Bungee library target: "bungee_static"
file(GLOB BUNGEE_LIBRARY_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/bungee/*.h")
file(GLOB BUNGEE_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(bungee_static STATIC ${BUNGEE_LIBRARY_HEADERS} ${BUNGEE_SOURCE_FILES} ${CMAKE_CURRENT_BINARY_DIR}/version.cpp)
add_custom_command(OUTPUT version.cpp _ COMMAND ${CMAKE_COMMAND} -DVERSION_NAMESPACE=Bungee -DGIT_CWD=${CMAKE_CURRENT_SOURCE_DIR} -DVERSION_CPP=version.cpp -P ${CMAKE_CURRENT_SOURCE_DIR}/version.cmake)
target_include_directories(bungee_static PRIVATE  submodules/eigen submodules .)
target_compile_definitions(bungee_static PRIVATE
  BUNGEE_SELF_TEST=${BUNGEE_SELF_TEST}
  eigen_assert=BUNGEE_ASSERT1
  EIGEN_DONT_PARALLELIZE=1
  BUNGEE_VISIBILITY=${BUNGEE_EXPORT}
)
target_compile_options(bungee_static PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fwrapv>)
target_link_libraries(bungee_static PRIVATE kissfft)

# Bungee library target: "bungee_shared"
if (BUNGEE_BUILD_SHARED_LIBRARY)
  add_library(bungee_shared SHARED $<TARGET_PROPERTY:bungee_static,SOURCES>)
  target_include_directories(bungee_shared PRIVATE $<TARGET_PROPERTY:bungee_static,INCLUDE_DIRECTORIES>)
  target_compile_options(bungee_shared PRIVATE $<TARGET_PROPERTY:bungee_static,COMPILE_OPTIONS>)
  target_link_libraries(bungee_shared PRIVATE $<TARGET_PROPERTY:bungee_static,LINK_LIBRARIES>)
  target_compile_definitions(bungee_shared PRIVATE $<TARGET_PROPERTY:bungee_static,COMPILE_DEFINITIONS>)
  set_target_properties(bungee_shared PROPERTIES OUTPUT_NAME bungee)
  add_dependencies(bungee_shared bungee_static) # to make Xcode happy
endif()

# Bungee command-line executable target: "bungee_executable"
add_executable(bungee_executable cmd/main.cpp)
if (BUNGEE_BUILD_SHARED_LIBRARY)
  target_link_libraries(bungee_executable PRIVATE bungee_shared)
else()
  target_link_libraries(bungee_executable PRIVATE bungee_static)
endif()
target_include_directories(bungee_executable PRIVATE submodules/cxxopts/include .)
set_target_properties(bungee_executable PROPERTIES OUTPUT_NAME bungee)

# KissFFT build
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
set(KISSFFT_PKGCONFIG OFF CACHE INTERNAL "" FORCE)
set(KISSFFT_STATIC ON CACHE INTERNAL "" FORCE)
set(KISSFFT_TEST OFF CACHE INTERNAL "" FORCE)
set(KISSFFT_TOOLS OFF CACHE INTERNAL "" FORCE)
add_subdirectory(submodules/kissfft EXCLUDE_FROM_ALL) # -Wcast-align

# Installation
if (BUNGEE_BUILD_SHARED_LIBRARY)
  if (APPLE)
    set_target_properties(bungee_shared PROPERTIES
      FRAMEWORK TRUE
      XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "iPhone Developer"
      MACOSX_FRAMEWORK_BUNDLE_VERSION ${BUNGEE_VERSION}
      MACOSX_FRAMEWORK_SHORT_VERSION_STRING ${BUNGEE_VERSION}
      VERSION ${BUNGEE_VERSION}
    )
    set_source_files_properties(
      ${BUNGEE_LIBRARY_HEADERS} 
      PROPERTIES
      MACOSX_PACKAGE_LOCATION Headers/bungee
    )
  else()
    install(FILES 
      ${BUNGEE_LIBRARY_HEADERS}
      DESTINATION ${CMAKE_INSTALL_PREFIX}/include/bungee
    )
  endif()
  install(TARGETS bungee_shared
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/library
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/library
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_PREFIX}
  )
  install(FILES 
    cmd/main.cpp  
    submodules/cxxopts/include/cxxopts.hpp
    DESTINATION ${CMAKE_INSTALL_PREFIX}/sample
  )
endif()
install(TARGETS bungee_executable
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime
  BUNDLE DESTINATION ${CMAKE_INSTALL_PREFIX}
)

