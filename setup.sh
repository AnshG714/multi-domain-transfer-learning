#!/bin/bash

print_usage() {
  echo "Use this script with either no flags or -s to skip downloads and unzipping."
}

s_flag=false

while getopts 's' flag; do
  case "${flag}" in
    s) s_flag=true ;;
    *) print_usage
       exit 1;;
  esac
done

# make the datasets directory if it doesn't exist.
DATASET_DIR='./datasets'
FLOWERS_DIR="$DATASET_DIR/flowers"
CARS_DIR="$DATASET_DIR/cars"
DOGS_DIR="$DATASET_DIR/dogs"

if [ ! -d "$DATASET_DIR" ]; then
  mkdir $DATASET_DIR
  mkdir $FLOWERS_DIR
  mkdir $CARS_DIR
  mkdir $DOGS_DIR
fi

# ------------- set up the flowers dataset --------------
if [[ "$s_flag" = "false" ]]; then
  wget https://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/102flowers.tgz -P $FLOWERS_DIR
  wget https://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/imagelabels.mat -P $FLOWERS_DIR
  wget https://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/setid.mat -P $FLOWERS_DIR
fi

tar -xzf "$FLOWERS_DIR/102flowers.tgz" -C $FLOWERS_DIR
mv "$FLOWERS_DIR/jpg" "$FLOWERS_DIR/images" 

# -------------- set up the cars dataset ------------------
if [[ "$s_flag" = "false" ]]; then
  wget http://ai.stanford.edu/~jkrause/car196/cars_train.tgz -P $CARS_DIR
  wget http://ai.stanford.edu/~jkrause/car196/cars_test.tgz -P $CARS_DIR
  wget https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz -P $CARS_DIR 
fi

tar -xzf "$CARS_DIR/car_devkit.tgz" -C $CARS_DIR
tar -xzf "$CARS_DIR/cars_test.tgz" -C $CARS_DIR
tar -xzf "$CARS_DIR/cars_train.tgz" -C $CARS_DIR

wget http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat -P "$CARS_DIR/devkit"
rm "$CARS_DIR/devkit/cars_test_annos.mat"

# -------------- set up the dogs dataset ------------------
wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar -P $DOGS_DIR
wget http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar -P $DOGS_DIR

tar -xvf "$DOGS_DIR/lists.tar" -C $DOGS_DIR
tar -xvf "$DOGS_DIR/images.tar" -C $DOGS_DIR

# ---------- Run utility script to set up annotations for pytorch ----------
python ./utils/create-dataset.py



