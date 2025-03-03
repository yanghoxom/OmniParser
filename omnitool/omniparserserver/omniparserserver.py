'''
python -m omniparserserver --som_model_path ../../weights/icon_detect/model.pt --caption_model_name florence2 --caption_model_path ../../weights/icon_caption_florence --device cuda --BOX_TRESHOLD 0.05 --num_cores 32 --batch_size 256
'''

import sys
import os
import time
import multiprocessing
from fastapi import FastAPI
from pydantic import BaseModel
import argparse
import uvicorn
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from util.omniparser import Omniparser

def parse_arguments():
    parser = argparse.ArgumentParser(description='Omniparser API')
    parser.add_argument('--som_model_path', type=str, default='../../weights/icon_detect/model.pt', help='Path to the som model')
    parser.add_argument('--caption_model_name', type=str, default='florence2', help='Name of the caption model')
    parser.add_argument('--caption_model_path', type=str, default='../../weights/icon_caption_florence', help='Path to the caption model')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model')
    parser.add_argument('--BOX_TRESHOLD', type=float, default=0.05, help='Threshold for box detection')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for the API')
    parser.add_argument('--port', type=int, default=8000, help='Port for the API')
    # Thêm tham số cho đa luồng và tối ưu
    parser.add_argument('--num_cores', type=int, default=multiprocessing.cpu_count(), help='Number of CPU cores to use')
    parser.add_argument('--max_workers', type=int, default=None, help='Maximum number of workers for thread pool')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for processing')
    args = parser.parse_args()
    return args

args = parse_arguments()
config = vars(args)

app = FastAPI()
omniparser = Omniparser(config)

class ParseRequest(BaseModel):
    base64_image: str

@app.post("/parse/")
async def parse(parse_request: ParseRequest):
    print('start parsing...')
    start = time.time()
    dino_labled_img, parsed_content_list = omniparser.parse(parse_request.base64_image)
    latency = time.time() - start
    print('time:', latency)
    return {"som_image_base64": dino_labled_img, "parsed_content_list": parsed_content_list, 'latency': latency}

@app.get("/probe/")
async def root():
    return {"message": "Omniparser API ready"}

@app.get("/info/")
async def info():
    """Trả về thông tin về cấu hình hiện tại của server"""
    return {
        "cores_configured": omniparser.num_cores,
        "max_workers": omniparser.max_workers,
        "batch_size": omniparser.optimal_batch_size,
        "device": omniparser.config.get('device', 'cpu'),
    }

if __name__ == "__main__":
    print(f"Starting server with {omniparser.num_cores} cores")
    uvicorn.run("omniparserserver:app", host=args.host, port=args.port, reload=True)