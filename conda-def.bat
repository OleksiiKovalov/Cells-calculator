rem conda-pack -n cellcounter20250525  -o cellcounter20250525.tar.gz
conda create --yes -n cellcounter20250525 python=3.11
conda activate cellcounter20250525
pip install -r Req_CellPose_InstanSeg_Stardist_yolo_20250525.txt
