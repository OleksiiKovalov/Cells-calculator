"""
Model conversion utility for InstanSeg.

Converts InstanSeg PyTorch model weights to TorchScript format for optimized
inference. This is a one-off script to prepare trained models for deployment.

Usage:
    1. Place model weights and experiment_log.csv in instanseg/<model_name>
    2. Update model_name variable at top of script
    3. Run: python convert_instanseg_model.py
    4. Converted model will be saved to trainedmodels/
    5. Register model in modelconfig.json
"""

# set model name
model_name = "Instanseg-Neuroblastoma-v4.pth"
# place new files (experiment_log.csv and model_weights_best.pth) under Cells-calculator\instanseg\<model_name> folder
# run this script python convert_instanseg_model.py
# you will get model_name.pt file (eg instanseg_20250602.pth.pt)  
# in the Cells-calculator\trainedmodels folder - register it in Cells-calculator\modelconfig.json config file


import os
os.environ["INSTANSEG_MODEL_PATH"] = str(os.path.abspath("../instanseg"))
os.environ["INSTANSEG_TORCHSCRIPT_PATH"] = str(os.path.abspath("../trainedmodels"))


from instanseg.utils.utils import export_to_torchscript
import torch


dummy = torch.randn(1, 3, 512, 512)  
export_to_torchscript(model_name)
# instanseg_script = torch.jit.load(os.path.join(os.environ["INSTANSEG_TORCHSCRIPT_PATH"],model_name + ".pt"))


# Then you can use the model for inference
# from instanseg.inference_class import InstanSeg
# instanseg_inference_class = InstanSeg(instanseg_script)