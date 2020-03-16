export LC_ALL=C
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

echo "update everything"
sudo apt-get update

echo "installing dependencies"
sudo apt-get install -y git python3-dev python3-pip
sudo apt-get install -y python3-tk


pip3 install ipython
pip3 install matplotlib
sudo apt-get install -y curl
sudo apt-get install -y libglew-dev
# dart dependencies
sudo apt-get install -y build-essential cmake pkg-config
sudo apt-get install -y libeigen3-dev libassimp-dev libccd-dev libfcl-dev libboost-regex-dev libboost-system-dev
sudo apt-get install -y libopenscenegraph-dev
sudo apt-get install -y libnlopt-dev
sudo apt-get install -y coinor-libipopt-dev
sudo apt-get install -y libbullet-dev
sudo apt-get install -y libode-dev
sudo apt-get install -y liboctomap-dev
sudo apt-get install -y libflann-dev
sudo apt-get install -y libtinyxml2-dev
sudo apt-get install -y liburdfdom-dev
sudo apt-get install -y libxi-dev libxmu-dev freeglut3-dev
sudo apt-get install -y libopenscenegraph-dev

echo "downloading boost 1.66.0"
wget https://dl.bintray.com/boostorg/release/1.66.0/source/boost_1_66_0.tar.gz
tar -xzf ./boost_1_66_0.tar.gz
rm ./boost_1_66_0.tar.gz

echo "installing boost"
cd ./boost_1_66_0
./bootstrap.sh --with-python=python3 --with-libraries=atomic,chrono,filesystem,python,system,regex
sudo ./b2 --with-python --with-filesystem --with-regex --with-system install
cd ..

echo "installing tensorflow"
pip3 install --upgrade pip
pip3 install tensorflow

echo "cloning dart"
git clone https://github.com/dartsim/dart.git

echo "installing dart"
cd dart
git checkout tags/v6.7.0

mkdir build
cd build
cmake ..
make -j4
sudo make install
cd ../..

echo "cloning ICC"
git clone https://github.com/snumrl/ICC.git

echo "installing ICC"
cd ICC
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
cd ..
