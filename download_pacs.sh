FILE_COR="./dataset/PACS"

# download tiny-imagenet-c
if [[ -d "$FILE_COR" ]]; then
  echo "$FILE_COR exists."
else
  echo "$FILE_COR does not exist. Start downloading..."
  if [[ ! -d "./dataset" ]]; then
    mkdir dataset
  fi
  python download_pacs.py
fi