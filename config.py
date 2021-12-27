from pydantic import BaseSettings


class Settings(BaseSettings):
    config_path_0_5m: str = "work_dirs/configs/faster_rcnn_orpn_r50_fpn_1x_3m_giuabien.py"
    cpkt_path_0_5m: str = "work_dirs/faster_rcnn_orpn_r50_fpn_1x_3m_giuabien/epoch_6.pth"
    split_cfg_0_5m: str = "work_dirs/split_config/ss_test_3m_giuabien.json"

    config_path_3m: str = "work_dirs/configs/faster_rcnn_orpn_r50_fpn_1x_3m_giuabien.py"
    cpkt_path_3m: str = 'work_dirs/faster_rcnn_orpn_r50_fpn_1x_3m_giuabien/epoch_6.pth'
    split_cfg_3m: str = "work_dirs/split_config/ss_test_3m_giuabien.json"

    device: str = "cuda:0"

    score_thr: float = 0.5

    class Config:
        env_file = ".env"
