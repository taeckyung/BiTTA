FILE_COR="./dataset/Tiny-ImageNet-C.tar"

# download tiny-imagenet-c
if [[ -f "$FILE_COR" ]]; then
  echo "$FILE_COR exists."
else
  echo "$FILE_COR does not exist. Start downloading..."
  if [[ ! -d "./dataset" ]]; then
    mkdir dataset
  fi
  cd dataset
  wget https://zenodo.org/records/2536630/files/Tiny-ImageNet-C.tar
fi

cd dataset
# unzip downloaded files
tar -xvf Tiny-ImageNet-C.tar