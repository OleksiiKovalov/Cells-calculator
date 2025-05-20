# import os
# os.environ["INSTANSEG_MODEL_PATH"] = str(os.path.abspath("../instanseg/models/"))
# os.environ["INSTANSEG_TORCHSCRIPT_PATH"] = str(os.path.abspath("../instanseg/torchscripts/"))


# from instanseg.utils.utils import export_to_torchscript
# import torch

# model_name = "instanseg_model_weights_best.pth"
# export_to_torchscript(model_name, show_example=True)
# instanseg_script = torch.jit.load(os.path.join(os.environ["INSTANSEG_TORCHSCRIPT_PATH"],model_name + ".pt"))


# #Then you can use the model for inference
# from instanseg.inference_class import InstanSeg
# instanseg_inference_class = InstanSeg(instanseg_script)