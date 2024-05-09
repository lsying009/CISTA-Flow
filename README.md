# CISTA-Flow

This repository is the official implementation of [Enhanced Event-Based Video Reconstruction with Motion Compensation](https://arxiv.org/pdf/2403.11961). In this work, we introduce a CISTA-Flow network to enhance reconstruction with motion compensation by integrating [CISTA-LSTC](https://ieeexplore.ieee.org/abstract/document/10130595) with a flow estimation network, particularly utilizing the [DCEIFlow](https://github.com/danqu130/DCEIFlow) or [ERAFT](https://github.com/uzh-rpg/E-RAFT). In this model, the reconstructed image and corresponding events are used to estimate flow. Subsequently, the flow is utilized to warp both the previously reconstructed frame and sparse codes as the inputs of CISTA-LSTC for current reconstruction.

![arch](cista-flow-arch.pdf)
<embed src="cista-flow-arch.pdf" type="application/pdf" width="100%" height="600px" />



## Training
We propose an iterative training framework for this combined system, as illustrated in figure below. Note that ```train.py``` only contains code for training CISTA-Flow after obtaining DCEIFlow (GT I) and CISTA (GT Flow). ```path_to_flownet``` refers to the path to DCEIFlow (GT I) and ```path_to_e2v``` refers to the path to CISTA (GT Flow). We provided the two pretrained models ```dceiflow-GTI.pth.tar/eraft-GTI.pth.tar``` and ```cista-GTFlow.pth.tar``` under ```pretrained```. Here, ```model_mode="cista-eiflow" or "cista-eraft"```. 

![train](cista-flow-train.pdf)

```bash
python train.py \
--path_to_train_data $path_to_train_data \
--model_mode "cista-eiflow" \
--path_to_flownet pretrained/dceiflow-GTI.pth.tar \
--path_to_e2v pretrained/cista-GTFlow.pth.tar \
--batch_size 2 \
--epochs 45 \
--flow_epoch 20 \
--rec_epoch 5 \
--lr 1e-4 \
--num_events 15000 \
--is_SummaryWriter \
--load_epoch_for_train 0 \
--warp_mode 'forward' \
--image_dim 180 240 \
```

## Evaluation

We provide codes to evaluate reconstruction quality of our CISTA network family using [EVREAL](https://github.com/ercanburak/EVREAL) library. Please refer to [CISTA-EVREAL](https://github.com/lsying009/CISTA-EVREAL)

Here, we also provide codes for evaluating simulated and real datasets. ```test_with_flow.py``` is used if GT flow is available for simulated dataset. ```test_wo_flow.py``` is used if GT flow is not available, e.g. ECD and HQF datasets. ```test_noeval.py``` is used when no GT frames and flows are available, e.g. HS-ERGB dataset. ```test_mvsec.py``` is used for MVSEC dataset.

```test_data_mode='real'``` for real data sequences, and ```test_data_mode='upsampled'``` for simulated data sequences. ```dataset=SIM/ECD/HQF/MVSEC```. The pretrained CISTA-Flow networks are stored in the files ```pretrained/cista-eiflow.pth.tar``` and ```pretrained/cista-eraft.pth.tar```.

```bash
python test_with_flow.py \
--path_to_test_model pretrained/cista-eiflow.pth.tar \
--path_to_test_data $path_to_test_data \
--model_mode "cista-eiflow" \
--test_data_mode 'upsampled' \
--dataset SIM \
--num_events -1 \
--is_load_flow \
--image_dim ${image_dim[@]} \

python test_wo_flow.py \
--path_to_test_model pretrained/cista-eiflow.pth.tar \
--path_to_test_data $path_to_test_data \
--model_mode "cista-eiflow" \
--test_data_mode 'real' \
--dataset ECD \
--num_events $num_events \
--image_dim ${image_dim[@]} \
--test_data_name $test_data_name \
```

## Datasets

We train and test our networks on simulated dataset. We provide codes for generating simulated dataset based on video-to-events generation:
- [V2E_generation](https://github.com/lsying009/V2E_Generation)

Evaluation can also work on real dataset [HQF](https://timostoff.github.io/20ecnn), [ECD](https://rpg.ifi.uzh.ch/davis_data.html), [HS-ERGB](https://rpg.ifi.uzh.ch/TimeLens.html) and [MVSEC](https://daniilidis-group.github.io/mvsec/) data sequences.

## Citation
If you use any of this code, please cite the publication as follows:

```bibtex
@misc{liu2024enhanced,
title={Enhanced Event-Based Video Reconstruction with Motion Compensation}, 
author={Siying Liu and Pier Luigi Dragotti},
year={2024},
eprint={2403.11961},
archivePrefix={arXiv},
primaryClass={cs.CV}
}
```