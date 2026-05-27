macro(baba package)
find_package(${package})

if (${${package}_FOUND})
  message(STATUS "${package} found: ${${package}_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "${package} not found")
endif()
endmacro()