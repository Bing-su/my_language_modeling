from huggingface_hub import HfApi
from loguru import logger

if __name__ == "__main__":
    api = HfApi()

    file = "/workspace/mlm_modeling/save/converted_model/pytorch_model.bin"
    # file = "/workspace/mlm_modeling/save/my_model/pytorch_model.bin"
    logger.info(f"Upload {file}")
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo="pytorch_model.bin",
        repo_id="Bingsu/mobilebert_ko_mlm_1",
        repo_type="model",
    )
    logger.info("Done")
