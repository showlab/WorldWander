<div align="center">
<h1>
WorldWander: Bridging Egocentric and Exocentric Worlds in Video Generation
</h1>

<p align="center">
    <span>
        <a href="https://arxiv.org/pdf/2511.22098" target="_blank"> 
        <img src='https://img.shields.io/badge/arXiv%202511.22098-WorldWander-red' alt='Paper PDF'></a> &emsp;  &emsp; 
    </span>
    <span> 
        <a href='https://lulupig12138.github.io/WorldWander' target="_blank">
        <img src='https://img.shields.io/badge/Project_Page-WorldWander-green' alt='Project Page'></a>  &emsp;  &emsp;
    </span>
    <span> 
        <a href='' target="_blank"> 
        <img src='https://img.shields.io/badge/Datasets-WorldWander-yellow' alt='Hugging Face'></a> &emsp;  &emsp;
    </span>
</p>

</div>

## 🎉 News
<pre>
• <strong>2026.05</strong>: 🔥 The EgoExo-Synthetic dataset is now released.
• <strong>2025.12</strong>: 🔥 Our paper, training code, and project page are released.
</pre>

## 🎬 Teaser
<b>TL;DR:</b> We propose WorldWander, an in-context learning framework for translating between egocentric and exocentric worlds in video generation. We also release [EgoExo-8K](https://huggingface.co/datasets/QuanjianSong/EgoExo-Synthetic), a high-quality dataset containing synchronized egocentric–exocentric triplets. The teaser is shown below:
![Overall Framework](assets/teaser.png)



## 📖 Overview
Video diffusion models have recently achieved remarkable progress in realism and controllability. However, achieving seamless video translation across different perspectives, such as first-person (egocentric) and third-person (exocentric), remains underexplored. Bridging these perspectives is crucial for filmmaking, embodied AI, and world models.
Motivated by this, we present <b>WorldWander</b>, an in-context learning framework tailored for translating between egocentric and exocentric worlds in video generation. Building upon advanced video diffusion transformers, WorldWander integrates (i) <i>In-Context Perspective Alignment</i> and (ii) <i>Collaborative Position Encoding</i> to efficiently model cross-view synchronization.
Overall framework is shown below:
![Overall Framework](assets/overall_pipeline.png)



## 🤗 Datasets
To further support our task, we curate [EgoExo-8K](https://huggingface.co/datasets/QuanjianSong/EgoExo-Synthetic), a high-quality dataset containing synchronized egocentric–exocentric triplets from both <i>synthetic</i> and <i>real-world</i> scenarios.
We show some examples below:
![Datasets Example](assets/datasets_example.png)



## 🔧 Environment
```
git clone https://github.com/showlab/WorldWander.git
# Installation with the requirement.txt
conda create -n WorldWander python=3.10
conda activate WorldWander
pip install -r requirements.txt
# Installation with environment.yml
conda env create -f environment.yml
conda activate WorldWander
```



## 🚀 Try Inference
WorldWander is trained on the [wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) model using 4 H200 GPUs, with a batch size of 4 per GPU.
To make it easier for you to use directly, we provide the following checkpoints for different tasks:

| Models                             | Links                 | configs                                      |
| ---------------------------------- | --------------------- | ------------------------------------------- |
| wan2.2-TI2V-5B_three2one_synthetic | 🤗 [Huggingface](xxx) | configs/wan2-2_lora_three2one_synthetic.yaml |
| wan2.2-TI2V-5B_one2three_synthetic | 🤗 [Huggingface](xxx) | configs/wan2-2_lora_one2three_synthetic.yaml |
| wan2.2-TI2V-5B_three2one_realworld | 🤗 [Huggingface](xxx) | configs/wan2-2_lora_three2one_realworld.yaml |
| wan2.2-TI2V-5B_one2three_realworld | 🤗 [Huggingface](xxx) | configs/wan2-2_lora_one2three_realworld.yaml |

You can download the specific checkpoint above and specify the corresponding config file for inference.
For convenience, we have provided the following example script:
```
bash scripts/inference_wan2.sh
```
Note that the parameter `ckpt_path` needs to be updated to the path of the checkpoint you downloaded.
<b>It is recommended to run this code on a GPU with 80GB of VRAM to avoid out of memory.</b>



## 🔥 Custom Training
You can also train on your custom dataset. To achieve this, you first need to adjust the `first_video_root`, `third_video_root`, `ref_image_root`, and other parameters in corresponding `config` file. If necessary, you may need to modify the `CustomTrainDataset` class in `dataset/custom_dataset.py` according to the attributes of your own dataset.
For convenience, we have also provided the following training script:
```
bash scripts/train_wan2.sh
```

## 🤝 Acknowledgements
🙏 This codebase borrows parts from [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) and the [Wan2.2](https://github.com/Wan-Video/Wan2.2). Many thanks to them for their open-source contributions. I also want to thank my co–first author for his trust and support; and to anonymously thank the senior who taught me PyTorch Lightning, enabling me to build training code from scratch on my own.


## 🎓 Bibtex
👋 If you find this code useful for your research, we would appreciate it if you could cite:
```
@article{song2025worldwander,
  title={WorldWander: Bridging Egocentric and Exocentric Worlds in Video Generation},
  author={Song, Quanjian and Song, Yiren and Peng, Kelly and Gao, Yuan and Shou, Mike Zheng},
  journal={arXiv preprint arXiv:2511.22098},
  year={2025}
}
```
