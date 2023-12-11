from pathlib import Path
import gdown


CHECKPOINT_LINK = "https://drive.google.com/u/0/uc?id=1TQAx-Vsc_UdqlLLZ2dptA-8PgBBcB-3h&export=download"
SAVE_PATH = Path("test_model/")


def main():
    SAVE_PATH.mkdir(exist_ok=True, parents=True)
    gdown.download(CHECKPOINT_LINK, str(SAVE_PATH / "as-checkpoint.pth"))


if __name__ == "__main__":
    main()
