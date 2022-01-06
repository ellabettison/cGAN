DIR=$1
TARGET=$2
echo $DIR
echo $TARGET
i=0
for img in $DIR/*
do
  python3 edge_detection.py --i "$img" --o $TARGET/$i --s 512 --a 8
  ((i=i+1))
done