from blueprint import api_bp
from flask import request, jsonify

from container import Container
from ekyc.service.ekyc_service import EKYCService
import tempfile
from dependency_injector.wiring import Provide, inject

@api_bp.route("/ekyc-process", methods=["POST"])
@inject
def ekyc_verify(ekyc_service: EKYCService = Provide[Container.ekyc_service]):

    video_file = request.files.get("videoClip")
    cccd_front = request.files.get("frontID")
    cccd_back = request.files.get("backID")

    if not video_file or not cccd_front:
        return jsonify({"error": "Thiếu video hoặc cccd"}), 400
    

    with    tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video, \
            tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_cccd, \
            tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_cccd_back:

        video_file.save(tmp_video.name)
        cccd_front.save(tmp_cccd.name)
        cccd_back.save(tmp_cccd_back.name)

    ekyc_check = ekyc_service.process_ekyc(tmp_video.name, tmp_cccd.name)
    """
        ekyc_check = {
            "liveness": True,
            "face_matching": True
        }
    """
    ocr_data = ekyc_service.process_ocr(tmp_cccd.name)
    ocr_back = ekyc_service.process_ocr_back(tmp_cccd_back.name)
    """
        {
            "data": {
                "Số CCCD": "001234567890",
                "Họ và tên": "Nguyễn Văn A",
                "Ngày sinh": "01/01/1990",
                "Giới tính": "Nam",
                "Quốc tịch": "Việt Nam",
                "Quê quán": "Hà Nội",
                "Nơi thường trú": "TP Hồ Chí Minh, Quận 1"
            }
        }
    """

    return jsonify({
        "results": ekyc_check,
        "ocr": ocr_data,
        "ocr_back": ocr_back,
    })
