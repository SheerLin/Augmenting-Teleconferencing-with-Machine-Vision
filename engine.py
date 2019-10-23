import extractor

class Engine:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.frame = 0
        self.extractor = extractor.Extractor({
            'width': self.width,
            'height': self.height,
            'freq': 10,
            'closeness': 20
        })

    def process(self, im):
        # print(self.frame)
        
        im = self.extractor(im, self.frame)
        
        self.frame += 1
        return im


####################
# begin denoising
####################


# b,g,r = cv2.split(src)           # get b,g,r
# rgb_img = cv2.merge([r,g,b])     # switch it to rgb

# Denoising
# dst = cv2.fastNlMeansDenoisingColored(src,None,10,10,7,21)
# dst = cv2.medianBlur(src, 3)

# b,g,r = cv2.split(dst)           # get b,g,r
# rgb_dst = cv2.merge([r,g,b])     # switch it to rgb
# cv2.imshow("a", dst)

# fastNlMeansDenoisingColoredMulti()
