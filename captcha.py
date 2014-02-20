# -*- coding: utf-8 -*-
import random
import string

from PIL import Image, ImageDraw, ImageFont

def get_str(length=5, chars=string.ascii_letters + string.digits):
    return "".join(random.choice(chars) for i in xrange(length))

def get_captcha(text=None, chars=string.ascii_letters + string.digits):
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 110)
    text = text or get_str(chars=chars)
    size = font.getsize(text)
    img = Image.new("RGB", size=(size[0] + 10, size[1] + 20), color="#FFF")
    draw = ImageDraw.Draw(img)
    draw.text([5, 10], text, "#000", font=font)
    return img


if __name__ == '__main__':
   img = get_captcha()
   img.save("captcha.png")




