from pydantic import BaseSettings


class Settings(BaseSettings):
    config_05m_bien: str = "work_dirs/config/05m_bien.py"
    cpkt_05m_bien: str = "work_dirs/cpkt/05m_bien.pth"
    split_config_05m_bien: str = "work_dirs/split_config/05m_bien.json"

    config_05m_cang: str = "work_dirs/config/05m_cang.py"
    cpkt_05m_cang: str = "work_dirs/cpkt/05m_cang.pth"
    split_config_05m_cang: str = "work_dirs/split_config/05m_cang.json"

    config_05m_dao: str = "work_dirs/config/05m_dao.py"
    cpkt_05m_dao: str = "work_dirs/cpkt/05m_dao.pth"
    split_config_05m_dao: str = "work_dirs/split_config/05m_dao.json"

    config_3m_bien: str = "work_dirs/config/3m_bien.py"
    cpkt_3m_bien: str = "work_dirs/cpkt/3m_bien.pth"
    split_config_3m_bien: str = "work_dirs/split_config/3m_bien.json"

    config_3m_cang: str = "work_dirs/config/3m_cang.py"
    cpkt_3m_cang: str = "work_dirs/cpkt/3m_cang.pth"
    split_config_3m_cang: str = "work_dirs/split_config/3m_cang.json"

    config_3m_dao: str = "work_dirs/config/3m_dao.py"
    cpkt_3m_dao: str = "work_dirs/cpkt/3m_dao.pth"
    split_config_3m_dao: str = "work_dirs/split_config/3m_dao.json"

    device: str = "cuda:0"

    score_thr: float = 0.5

    class Config:
        env_file = ".env"
