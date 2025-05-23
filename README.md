# Text-to-3D Generation by 2D Editing
[Haoran Li](https://scholar.google.com/citations?user=Rxl8r70AAAAJ&hl=en), [Yuli Tian](https://github.com/lili174311), [Yonghui Wang](https://scholar.google.com.hk/citations?user=GGMWna4AAAAJ&hl=zh-CN), [Yong Liao](https://scholar.google.com/citations?user=_wuoU1EAAAAJ&hl=en), [Lin Wang](https://scholar.google.com/citations?user=SReb2csAAAAJ&hl=en), [Yuyang Wang](https://scholar.google.com/citations?user=D1HTbhEAAAAJ&hl=en), [Peng Yuan Zhou](https://scholar.google.com/citations?user=6n-ELeoAAAAJ&hl=en)

This repository contains the official implementation for [Text-to-3D Generation by 2D Editing](https://arxiv.org/pdf/2412.05929).

[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://jahnsonblack.github.io/GE3D/) [![arXiv](https://img.shields.io/badge/arXiv-2412.05929-b31b1b.svg)](https://arxiv.org/pdf/2412.05929)

Note: We compress these motion pictures for faster previewing. More high-quality 3D results can be found on our project homepage [GE3D](https://jahnsonblack.github.io/GE3D/).
 <table class="center">
    <tr style="line-height: 0">
      <td width=24% style="border: none; text-align: center">A wooden rocking chair, rustic, comfortable, 8K. </td>
      <td width=24% style="border: none; text-align: center">A fluffy squirrel wearing a tiny wizard hat.</td>
      <td width=24% style="border: none; text-align: center">Black Widow in Marvel, head, photorealistic, 8K, HDR.</td>
      <td width=24% style="border: none; text-align: center">Ninja in black outfit, photorealistic, 8K, HDR.</td>
    </tr>
    <tr style="line-height: 0">
      <td width=24% style="border: none"><img src="assets/wooden_rocking_chair.gif"></td>
      <td width=24% style="border: none"><img src="assets/squirrel.gif"></td>
      <td width=24% style="border: none"><img src="assets/BlackWidow.gif"></td>
      <td width=24% style="border: none"><img src="assets//Ninja.gif"></td>
    </tr>
 </table>


## Getting Start!
### Requirments

```bash
git clone https://github.com/Jahnsonblack/GE3D.git
cd GE3D

conda create -n ge3d python=3.10
conda activate ge3d

pip install -r requirements.txt -f https://download.pytorch.org/whl/cu118/torch_stable.html 

git clone --recursive https://github.com/Jahnsonblack/diff-gaussian-rasterization.git
git clone https://github.com/YixunLiang/simple-knn.git
pip install diff-gaussian-rasterization/
pip install simple-knn/


# Install point-e
git clone https://github.com/crockwell/Cap3D.git
cd Cap3D/text-to-3D/point-e/
pip install -e .
```

```sh
cd GE3D
mkdir point_e_model_cache
# Optional: Initialize with better point-e
wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/our_finetuned_models/pointE_finetuned_with_825kdata.pth
mv pointE_finetuned_with_825kdata.pth point_e_model_cache/
# Modify the parameter init_guided in the configuration file to pointe_825k
# or
wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/our_finetuned_models/pointE_finetuned_with_330kdata.pth
mv pointE_finetuned_with_330kdata.pth point_e_model_cache/
# Modify the parameter init_guided in the configuration file to pointe_330k
```

### Generate Single Object

```bash
python train.py --opt configs/sample.yaml
```
To achieve higher quality 3D generation results, you can increase the number of iterations and use smaller step sizes in the later stages of optimization.

## Acknowledgement
This work is built on many amazing research works and open-source projects :
- [LucidDreamer](https://github.com/EnVision-Research/LucidDreamer)
- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [Stable-Dreamfusion](https://github.com/ashawkey/stable-dreamfusion)
- [Point-E](https://github.com/openai/point-e)

Thanks for their excellent work and great contribution to 3D generation area.

## Citation
If you find it useful in your research, please consider citing our paper as follows:
```
@article{li2024text2d3d,
  title={Text-to-3D Generation by 2D Editing},
  author={Li, Haoran and Tian, Yuli and Wang, Yonghui and Liao, Yong and Wang, Lin and Wang, Yuyang and Zhou, Peng Yuan},
  journal={arXiv preprint arXiv:2412.05929},
  year={2024}
}


```
