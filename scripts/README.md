# Some additional scripts for pre-processing

 - To crop IJB-C:
    ```bash
    python crop_ijbc.py 
    /gpfs/gpfs0/r.karimov/trash/IJB/IJB-C/protocols/archive/ijbc_metadata.csv \
    /gpfs/gpfs0/r.karimov/trash/IJB/IJB-C/images \
    data/ijbc_cropped/
    ```
 - To align IJB-C:
    ```bash
   python mtcnn_align_ijb.py \
   --path data/ijbc_cropped/ \
   --save-prefix data/ijbc_aligned
    ```
   
 - To crop IJB-A (Note: not tested):
    ```bash
    python align/crop_ijba.py metadata/ijba_crop_metadata.csv \
    /path/to/IJB-A/images/ \
    data/ijba_cropped/
    ```


