try:
    from textcnn.model import TextCNN
except ImportError:  # TextCNN is not defined until you implement model.py
    TextCNN = None  # type: ignore[misc, assignment]

__all__ = ["TextCNN"]
