from util.utils import get_som_labeled_img, get_caption_model_processor, get_yolo_model, check_ocr_box
import torch
from PIL import Image
import io
import base64
from typing import Dict
import multiprocessing
import os
import concurrent.futures
from functools import partial

class Omniparser(object):
    def __init__(self, config: Dict):
        self.config = config
        device = config.get('device', 'cpu')

        # Cấu hình đa luồng tối ưu
        self.num_cores = config.get('num_cores', multiprocessing.cpu_count())
        torch.set_num_threads(self.num_cores)
        os.environ['OMP_NUM_THREADS'] = str(self.num_cores)
        os.environ['MKL_NUM_THREADS'] = str(self.num_cores)
        os.environ['OPENBLAS_NUM_THREADS'] = str(self.num_cores)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(self.num_cores)
        os.environ['NUMEXPR_NUM_THREADS'] = str(self.num_cores)

        # Thiết lập max workers cho thread pool
        self.max_workers = config.get('max_workers', min(32, self.num_cores * 4))

        # Số lượng batch tối ưu dựa trên số lượng CPU
        self.optimal_batch_size = config.get('batch_size', min(256, self.num_cores * 8))

        self.som_model = get_yolo_model(model_path=config['som_model_path'])
        self.caption_model_processor = get_caption_model_processor(
            model_name=config['caption_model_name'],
            model_name_or_path=config['caption_model_path'],
            device=device,
            num_threads=self.num_cores
        )
        print(f'Omniparser initialized with {self.num_cores} cores, {self.max_workers} workers, batch size: {self.optimal_batch_size}')

    def parse(self, image_base64: str):
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        print('image size:', image.size)

        box_overlay_ratio = max(image.size) / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        # Sử dụng ThreadPoolExecutor cho OCR và model inference
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Chạy OCR trong một thread riêng
            ocr_future = executor.submit(
                check_ocr_box,
                image,
                display_img=False,
                output_bb_format='xyxy',
                easyocr_args={'text_threshold': 0.8},
                use_paddleocr=False
            )

            # Lấy kết quả từ OCR
            (text, ocr_bbox), _ = ocr_future.result()

        # Sử dụng batch size tối ưu từ cấu hình
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image,
            self.som_model,
            BOX_TRESHOLD=self.config['BOX_TRESHOLD'],
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=self.caption_model_processor,
            ocr_text=text,
            use_local_semantics=True,
            iou_threshold=0.7,
            scale_img=False,
            batch_size=self.optimal_batch_size
        )

        return dino_labled_img, parsed_content_list