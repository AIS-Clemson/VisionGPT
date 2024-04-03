# VisionGPT: LLM-Assisted Real-Time Anomaly Detection for Safe Visual Navigation

## Overview
This project explores the potential of Large Language Models(LLMs) in zero-shot anomaly detection for safe visual navigation. 

<br>

With the assistance of the state-of-the-art real-time open-world
object detection model Yolo-World and specialized prompts, the proposed framework can identify anomalies within camera-captured frames that include any possible obstacles, then generate
concise, audio-delivered descriptions emphasizing abnormalities, assist in safe visual navigation
in complex circumstances. 


<div align="center">
    <img src="./Figures/frame.png" alt="Framework" style="width: 60%;">
</div>

Moreover, our proposed framework leverages the advantages of LLMs
and the open-vocabulary object detection model to achieve the dynamic scenario switch, which
allows users to transition smoothly from scene to scene, which addresses the limitation of traditional
visual navigation. 

Furthermore, this project explored the performance contribution of different prompt
components, provided the vision for future improvement in visual accessibility, and paved the way
for LLMs in video anomaly detection and vision-language understanding.


## Method

### Yolo-World
We apply the latest yolo world for the open-world object detection task to adapt the system in any scenario and any situation. 

### GPT-3.5
We apply GPT-3.5 for fast respond and low-cost.


### H-splitter
We implemented a H-splitter to assist object detection and categorize the objects into 3 different types based on the priority. 

See our another project for more info: <a href="https://github.com/JiayouQin/H-Splitter" target="_blank">H-Splitter</a>






Please cite our work if you find this project is helpful.

@article{wang2024visiongpt,
  title={VisionGPT: LLM-Assisted Real-Time Anomaly Detection for Safe Visual Navigation},
  author={Wang, Hao and Qin, Jiayou and Bastola, Ashish and Chen, Xiwen and Suchanek, John and Gong, Zihao and Razi, Abolfazl},
  journal={arXiv preprint arXiv:2403.12415},
  year={2024}
}

* This paper is under reviewing of iROS 2024.
