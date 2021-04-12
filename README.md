# Face library for huawei project

<em>Note: the codebase is still very buggy and messy</em>

### INSTALL

For now just install the requirements in `requirements.txt`

### Training SphereFace [[1]](#1) models:

```bash 
bash train_scripts/spherenet/run_spherenet_base.sh
```

#### and with PFE head:

```bash 
bash train_scripts/spherenet/run_spherenet_pfe.sh
```

### Training ArcFace [[2]](#2) models:

```bash 
bash train_scripts/arcface/run_arcface_base.sh
```

<em>Note: you need to reallocate 4 GPUs to train the ArcFace model. 
(Not tested on less GPUs) We could ease the requirement in the 
future though I'm not sure if it is possible to train it on single machine
because of the large number of output classes.</em>

### Aligned data

- [x] CasiaWeb Face dataset is aligned and is stored in the 
cluster (all the pathes are given in the models' configs)
  
- [x] LFW is aligned.

- [x] MS1M can be used for training. Don't know how it is aligend. (#TODO)

- [ ] IJB-A is not parsable. There are null images in the folder. 
  Waiting for the response from the authors.

- [ ] IJB-C

## References
<a id="1">[1]</a> 
Liu et at.,
SphereFace: Deep Hypersphere Embedding for Face Recognition. 
2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)

<a id="2">[2]</a> 
Deng et at.,
ArcFace: Additive Angular Margin Loss for Deep Face Recognition.
2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).