import hough

class Engine:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.frame = 0
        self.extractor = hough.Extractor({
            'width': self.width,
            'height': self.height,
            'freq': 10,
            'closeness': 20
        })

    def process(self, im):
        # print(self.frame)

        im, edg = self.extractor(im, self.frame)
        
        self.frame += 1
        return im, edg
