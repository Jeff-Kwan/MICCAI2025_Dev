'''
ONNX is 2x slower, not worth it.
'''

import torch
# import onnxruntime as ort
# import io
import json
from time import time
from monai.inferers import sliding_window_inference

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.AttnUNet2 import AttnUNet

torch_model = AttnUNet(json.load(open("configs/model/attn_unet.json")))
torch_model.eval()

trials = 1
B, C, H, W, D = 1, 1, 280, 280, 200
patch_size = (192, 192, 128)
overlap = 0.25
dummy_input = torch.randn(B, C, H, W, D)


# Time the native PyTorch + MONAI sliding window
start_time = time()
with torch.inference_mode():
    for i in range(trials):
        torch_output = sliding_window_inference(
            dummy_input,
            roi_size=patch_size,
            sw_batch_size=1,
            predictor=lambda x: torch_model(x),
            overlap=overlap,
            mode="gaussian",
            buffer_steps=1,
        ).squeeze(0)
end_time = time()
print(f"Torch model inference time: {(end_time - start_time)/trials:.4f} seconds")

# # ONNX model
# buffer = io.BytesIO()
# torch.onnx.export(
#     torch_model,
#     torch.randn(B, C, *patch_size),  # dummy input for export
#     buffer,                              # <-- file-like buffer
#     input_names=["input"],
#     output_names=["output"],
# )
# onnx_bytes = buffer.getvalue()           # grab the raw ONNX protobuf

# # 2) Create your ONNX Runtime session from those bytes
# session = ort.InferenceSession(
#     onnx_bytes,                          # <-- feed in-memory model
#     providers=["CPUExecutionProvider"]
# )
# def onnx_predictor(x: torch.Tensor) -> torch.Tensor:
#     # x is a torch.Tensor of shape [1, C, ph, pw, pd]
#     x_np = x.cpu().numpy()
#     input_name = session.get_inputs()[0].name
#     ort_outs = session.run(None, {input_name: x_np})
#     # assume single output
#     out_np = ort_outs[0]
#     return torch.from_numpy(out_np)

# # Time the ONNX + MONAI sliding window
# start_time = time()
# for i in range(trials):
#     onnx_output = sliding_window_inference(
#         dummy_input,
#         roi_size=patch_size,
#         sw_batch_size=1,
#         predictor=onnx_predictor,
#         overlap=overlap,
#         mode="gaussian",
#         buffer_steps=1,
#     ).squeeze(0)
# end_time = time()
# print(f"ONNX model inference time: {(end_time - start_time)/trials:.4f} seconds")

# # Compare the outputs
# diff = torch.abs(torch_output - onnx_output)
# print(f"Max absolute difference: {diff.max():.6f}")
# print(f"Mean absolute difference: {diff.mean():.6f}")