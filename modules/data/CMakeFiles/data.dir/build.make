# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.2

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/adrian/Desktop/gpustore/sources/modules/data

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/adrian/Desktop/gpustore/sources/modules/data

# Include any dependencies generated for this target.
include CMakeFiles/data.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/data.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/data.dir/flags.make

CMakeFiles/data.dir/test/test.cpp.o: CMakeFiles/data.dir/flags.make
CMakeFiles/data.dir/test/test.cpp.o: test/test.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/adrian/Desktop/gpustore/sources/modules/data/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/data.dir/test/test.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/data.dir/test/test.cpp.o -c /home/adrian/Desktop/gpustore/sources/modules/data/test/test.cpp

CMakeFiles/data.dir/test/test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/data.dir/test/test.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/adrian/Desktop/gpustore/sources/modules/data/test/test.cpp > CMakeFiles/data.dir/test/test.cpp.i

CMakeFiles/data.dir/test/test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/data.dir/test/test.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/adrian/Desktop/gpustore/sources/modules/data/test/test.cpp -o CMakeFiles/data.dir/test/test.cpp.s

CMakeFiles/data.dir/test/test.cpp.o.requires:
.PHONY : CMakeFiles/data.dir/test/test.cpp.o.requires

CMakeFiles/data.dir/test/test.cpp.o.provides: CMakeFiles/data.dir/test/test.cpp.o.requires
	$(MAKE) -f CMakeFiles/data.dir/build.make CMakeFiles/data.dir/test/test.cpp.o.provides.build
.PHONY : CMakeFiles/data.dir/test/test.cpp.o.provides

CMakeFiles/data.dir/test/test.cpp.o.provides.build: CMakeFiles/data.dir/test/test.cpp.o

CMakeFiles/data.dir/src/file.cpp.o: CMakeFiles/data.dir/flags.make
CMakeFiles/data.dir/src/file.cpp.o: src/file.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/adrian/Desktop/gpustore/sources/modules/data/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/data.dir/src/file.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/data.dir/src/file.cpp.o -c /home/adrian/Desktop/gpustore/sources/modules/data/src/file.cpp

CMakeFiles/data.dir/src/file.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/data.dir/src/file.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/adrian/Desktop/gpustore/sources/modules/data/src/file.cpp > CMakeFiles/data.dir/src/file.cpp.i

CMakeFiles/data.dir/src/file.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/data.dir/src/file.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/adrian/Desktop/gpustore/sources/modules/data/src/file.cpp -o CMakeFiles/data.dir/src/file.cpp.s

CMakeFiles/data.dir/src/file.cpp.o.requires:
.PHONY : CMakeFiles/data.dir/src/file.cpp.o.requires

CMakeFiles/data.dir/src/file.cpp.o.provides: CMakeFiles/data.dir/src/file.cpp.o.requires
	$(MAKE) -f CMakeFiles/data.dir/build.make CMakeFiles/data.dir/src/file.cpp.o.provides.build
.PHONY : CMakeFiles/data.dir/src/file.cpp.o.provides

CMakeFiles/data.dir/src/file.cpp.o.provides.build: CMakeFiles/data.dir/src/file.cpp.o

CMakeFiles/data.dir/src/file_basics.cpp.o: CMakeFiles/data.dir/flags.make
CMakeFiles/data.dir/src/file_basics.cpp.o: src/file_basics.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/adrian/Desktop/gpustore/sources/modules/data/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/data.dir/src/file_basics.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/data.dir/src/file_basics.cpp.o -c /home/adrian/Desktop/gpustore/sources/modules/data/src/file_basics.cpp

CMakeFiles/data.dir/src/file_basics.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/data.dir/src/file_basics.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/adrian/Desktop/gpustore/sources/modules/data/src/file_basics.cpp > CMakeFiles/data.dir/src/file_basics.cpp.i

CMakeFiles/data.dir/src/file_basics.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/data.dir/src/file_basics.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/adrian/Desktop/gpustore/sources/modules/data/src/file_basics.cpp -o CMakeFiles/data.dir/src/file_basics.cpp.s

CMakeFiles/data.dir/src/file_basics.cpp.o.requires:
.PHONY : CMakeFiles/data.dir/src/file_basics.cpp.o.requires

CMakeFiles/data.dir/src/file_basics.cpp.o.provides: CMakeFiles/data.dir/src/file_basics.cpp.o.requires
	$(MAKE) -f CMakeFiles/data.dir/build.make CMakeFiles/data.dir/src/file_basics.cpp.o.provides.build
.PHONY : CMakeFiles/data.dir/src/file_basics.cpp.o.provides

CMakeFiles/data.dir/src/file_basics.cpp.o.provides.build: CMakeFiles/data.dir/src/file_basics.cpp.o

CMakeFiles/data.dir/src/time_series_reader.cpp.o: CMakeFiles/data.dir/flags.make
CMakeFiles/data.dir/src/time_series_reader.cpp.o: src/time_series_reader.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/adrian/Desktop/gpustore/sources/modules/data/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/data.dir/src/time_series_reader.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/data.dir/src/time_series_reader.cpp.o -c /home/adrian/Desktop/gpustore/sources/modules/data/src/time_series_reader.cpp

CMakeFiles/data.dir/src/time_series_reader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/data.dir/src/time_series_reader.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/adrian/Desktop/gpustore/sources/modules/data/src/time_series_reader.cpp > CMakeFiles/data.dir/src/time_series_reader.cpp.i

CMakeFiles/data.dir/src/time_series_reader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/data.dir/src/time_series_reader.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/adrian/Desktop/gpustore/sources/modules/data/src/time_series_reader.cpp -o CMakeFiles/data.dir/src/time_series_reader.cpp.s

CMakeFiles/data.dir/src/time_series_reader.cpp.o.requires:
.PHONY : CMakeFiles/data.dir/src/time_series_reader.cpp.o.requires

CMakeFiles/data.dir/src/time_series_reader.cpp.o.provides: CMakeFiles/data.dir/src/time_series_reader.cpp.o.requires
	$(MAKE) -f CMakeFiles/data.dir/build.make CMakeFiles/data.dir/src/time_series_reader.cpp.o.provides.build
.PHONY : CMakeFiles/data.dir/src/time_series_reader.cpp.o.provides

CMakeFiles/data.dir/src/time_series_reader.cpp.o.provides.build: CMakeFiles/data.dir/src/time_series_reader.cpp.o

# Object files for target data
data_OBJECTS = \
"CMakeFiles/data.dir/test/test.cpp.o" \
"CMakeFiles/data.dir/src/file.cpp.o" \
"CMakeFiles/data.dir/src/file_basics.cpp.o" \
"CMakeFiles/data.dir/src/time_series_reader.cpp.o"

# External object files for target data
data_EXTERNAL_OBJECTS =

lib/libdata.a: CMakeFiles/data.dir/test/test.cpp.o
lib/libdata.a: CMakeFiles/data.dir/src/file.cpp.o
lib/libdata.a: CMakeFiles/data.dir/src/file_basics.cpp.o
lib/libdata.a: CMakeFiles/data.dir/src/time_series_reader.cpp.o
lib/libdata.a: CMakeFiles/data.dir/build.make
lib/libdata.a: CMakeFiles/data.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX static library lib/libdata.a"
	$(CMAKE_COMMAND) -P CMakeFiles/data.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/data.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/data.dir/build: lib/libdata.a
.PHONY : CMakeFiles/data.dir/build

CMakeFiles/data.dir/requires: CMakeFiles/data.dir/test/test.cpp.o.requires
CMakeFiles/data.dir/requires: CMakeFiles/data.dir/src/file.cpp.o.requires
CMakeFiles/data.dir/requires: CMakeFiles/data.dir/src/file_basics.cpp.o.requires
CMakeFiles/data.dir/requires: CMakeFiles/data.dir/src/time_series_reader.cpp.o.requires
.PHONY : CMakeFiles/data.dir/requires

CMakeFiles/data.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/data.dir/cmake_clean.cmake
.PHONY : CMakeFiles/data.dir/clean

CMakeFiles/data.dir/depend:
	cd /home/adrian/Desktop/gpustore/sources/modules/data && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/adrian/Desktop/gpustore/sources/modules/data /home/adrian/Desktop/gpustore/sources/modules/data /home/adrian/Desktop/gpustore/sources/modules/data /home/adrian/Desktop/gpustore/sources/modules/data /home/adrian/Desktop/gpustore/sources/modules/data/CMakeFiles/data.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/data.dir/depend

