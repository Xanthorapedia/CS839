# CS839 Fall 2019
Welcome to my repo for CS839: Physics-Based Modeling and Simulation!

## Build
All projects in this repo can be built with Visual Studio Code 2019<sup>1</sup> on Windows. To build the project:
1. Set environment variable `USD_HOME` to be the root directory of your [USD](https://github.com/PixarAnimationStudios/USD) installation;
2. Execute the following<sup>2, 3</sup>:
```sh
$ mkdir build
$ cd build
$ cmake ..
$ cmake --build . --config Release
```
***1: For other toolsets and/or python/boost library versions, change [the `BOOST_PYTHON_LIB` attribute in CMakeLists.txt](CMakeLists.txt#L26) to the path to your boost_python-xxx-.lib***

*2: If you are building the projects with `Debug` option, the USD installation should also be built with `--debug` flag.*

*3: An alternative to setting the environment variable is to add `-DUSD_ROOT_DIR=<your_USD_installation>` in the `cmake` generation step.*


## Run
Executables can be found in `build`. Executes the executable to generate a `.usda` file in the working directory. Open it with USDView. E.g.:
```sh
$ ./src/Demos/Intro/helloWorld/Release/helloWorld
$ usdview helloWorld.usda
```
## Have fun!
