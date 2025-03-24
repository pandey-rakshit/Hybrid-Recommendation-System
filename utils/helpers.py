from pathlib import Path


def get_file_extension_from_path(input_path: str) -> str:
    """
    Extracts and returns the file extension from a given file path.

    Args:
        input_path (str): File path or URL.

    Returns:
        str: File extension in lowercase.
    """
    return Path(input_path).suffix.lstrip(".").lower()
