# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
import shutil
import os
import tempfile
import os
import torch
from correction.config import cfg
from correction.data.scalers import StandardScaler
import numpy as np


app = FastAPI()

@app.get("/")
async def read_item():
    with open("templates/index.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        result_file_path = process_file(tmp_path)

        return FileResponse(result_file_path, media_type="application/octet-stream", filename="output_data.npy")

    finally:
        os.unlink(tmp_path)

def process_file(file_path):
    # код для обработки файла
    result_file_path = './output_data.npy'


    Path_list = [[file_path]]

    input_data = torch.from_numpy(np.load(Path_list[0][0])) # [24,3,210,280]
    input_data = input_data[None,:] #added one dimensional
    PATH_model = r'model_best.pt'
    batch_size = 1

    wrf_scaler = StandardScaler()
    wrf_scaler.apply_scaler_channel_params(torch.load(r'C:\Users\PC\Desktop\хакатон\logs\wrf_means'),
                                           torch.load(r'C:\Users\PC\Desktop\хакатон\logs\wrf_stds'))

    model = torch.load(PATH_model)

    print(model)
    model.eval()

    input_data = torch.swapaxes(input_data.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
    input_data = wrf_scaler.channel_transform(input_data, 2)

    output = model(input_data)

    output = wrf_scaler.channel_inverse_transform(output, 2)
    output_data = output.cpu().detach().numpy()


    output_data = output.cpu().detach().numpy()  # move output tensor to CPU and convert to NumPy array

    np.save('output_data.npy', output_data)


    return 'output_data.npy'
