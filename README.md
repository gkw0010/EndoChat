# EndoChat: Grounded Multimodal Large Language Model for Endoscopic Surgery
Guankun Wang†, Long Bai†, Junyi Wang†, Kun Yuan†, Zhen Li, Tianxu Jiang, Xiting He, Jinlin Wu, Zhen Chen, Hongbin Liu, Nicolas Padoy, Nassir Navab, and Hongliang Ren* <br/>

## Overview
Recently, Multimodal Large Language Models (MLLMs) have demonstrated their immense potential in computer-aided diagnosis and decision-making. In the context of robotic-assisted surgery, MLLMs can serve as effective tools for surgical training and guidance. However, there is still a lack of MLLMs specialized for surgical scene understanding in clinical applications. In this work, we introduce EndoChat to address various dialogue paradigms and subtasks in surgical scene understanding that surgeons encounter. To train our EndoChat, we construct the Surg-396K dataset through a novel pipeline that systematically extracts surgical information and generates structured annotations based on collected large-scale endoscopic surgery datasets. Furthermore, we introduce a multi-scale visual token interaction mechanism and a visual contrast-based reasoning mechanism to enhance the model's representation learning and reasoning capabilities. Our model achieves state-of-the-art performance across five dialogue paradigms and eight surgical scene understanding tasks. Additionally, we conduct evaluations with professional surgeons, most of whom provide positive feedback on collaborating with EndoChat. Overall, these results demonstrate that our EndoChat has great potential to significantly advance training and automation in robotic-assisted surgery.
<p align="center">
  <img 
    width="1000"
    src="./figures/overview.png"
  >
</p>
