ARCH=$1

nsml run -d airush2 --shm-size 8G -a "--arch=$ARCH" -e main.py