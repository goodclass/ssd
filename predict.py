from PIL import Image
from ssd import SSD

# 检测图片
if __name__ == "__main__":
    ssd = SSD()

    image = Image.open("img_sample/street.jpg")
    r_image = ssd.detect_image(image)
    r_image.show()
