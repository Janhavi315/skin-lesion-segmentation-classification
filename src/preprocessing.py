
def image_to_array(image, size=(64, 64), normalize=True):
    """
    Convert a PIL image or NumPy image into a resized RGB array.
    """

    import numpy as np
    from PIL import Image

    # Convert the input into a PIL image.
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    elif not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL image or NumPy array.")

    # Ensure three colour channels.
    image = image.convert("RGB")

    # Resize to a consistent image size.
    image = image.resize(size, Image.Resampling.LANCZOS)

    # Convert the image into a NumPy array.
    image_array = np.asarray(image, dtype=np.float32)

    # Scale pixel values to the range 0 to 1.
    if normalize:
        image_array = image_array / 255.0

    return image_array
