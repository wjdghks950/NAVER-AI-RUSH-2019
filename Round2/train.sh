ARCH=$1

nsml run -d airush2 --memory 12G --shm-size 28G -a "--arch=$ARCH" -e main.py
