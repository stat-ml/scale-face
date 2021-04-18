# Some additional scripts for pre-processing

 - To crop IJB-C:
    ```bash
    python crop_ijbc.py 
    /gpfs/gpfs0/r.karimov/trash/IJB/IJB-C/protocols/archive/ijbc_metadata.csv \
    /gpfs/gpfs0/r.karimov/trash/IJB/IJB-C/images \
    data/ijbc_cropped/
    ```
   
 - To crop IJB-A (Note: not tested):
    ```bash
    python align/crop_ijba.py proto/IJB-A/metadata.csv \
    /path/to/IJB-A/images/ \
    data/ijba_cropped/
    ```


