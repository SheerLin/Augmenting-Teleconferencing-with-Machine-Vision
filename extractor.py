import cv2
import numpy as np

from reshape import four_point_transform

class Extractor:

    def __init__(self, params):
        self.points = None
        self.new_points = None
        self.new_seen = 0
        self.params = params
        self.area = self.params['width'] * self.params['height']

    def __call__(self, src, frame):
        return self.extract_whiteboard(src, frame)

    def pipelineA(self, src):
        orig = src.copy()

        # Lower the contrast
        # alpha 1  beta 0      --> no change
        # 0 < alpha < 1        --> lower contrast
        # alpha > 1            --> higher contrast
        # -127 < beta < +127   --> good range for brightness values
        alpha = 0.7
        beta = -10
        src_contrast = cv2.addWeighted(src, alpha, src, 0, beta)
        # cv2.imshow("Contrast", src_contrast)

        # Apply Color Filter
        src = src_contrast
        b = []
        g = []
        r = []
        for h in range(100, 140): # 480
            for w in range(300, 340): # 640
                b.append(src[h][w][0])
                g.append(src[h][w][1])
                r.append(src[h][w][2])
                for k in range(3):
                    src[h][w][k] = 255
        avg_b = sum(b)/len(b)
        avg_g = sum(g)/len(g)
        avg_r = sum(r)/len(r)
        delta = 35
        m = 1.5
        lower = np.array([avg_b-delta, avg_g-delta, avg_r-delta], dtype = "uint8")
        upper = np.array([avg_b+m*delta, avg_g+m*delta, avg_r+m*delta], dtype = "uint8")
        mask = cv2.inRange(src, lower, upper)
        src_white = cv2.bitwise_and(src, src, mask = mask)
        cv2.imshow('White', src_white)

        # Apply Morphing
        src = src_white
        morph = src
        kernel = np.ones((5, 5), np.uint8)
        iterations = 2
        morph = cv2.dilate(morph, kernel, iterations=iterations)
        morph = cv2.erode(morph, kernel, iterations=iterations)
        cv2.imshow('Open', morph)

        morph = cv2.erode(morph, kernel, iterations=iterations)
        morph = cv2.dilate(morph, kernel, iterations=iterations)
        cv2.imshow('Close', morph)

        # Convert the color from BGR to Gray
        src = morph
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Gray", src_gray)

        # Apply blurring to smoothen
        src = src_gray
        src_blur = cv2.GaussianBlur(src, (5, 5), 3, borderType=cv2.BORDER_ISOLATED)
        cv2.imshow("Blur", src_blur)

        # Apply mean adaptive theshold
        src = src_blur
        src_thresh = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 3)
        cv2.imshow("Thresh", src_thresh)

        _, src_thresh2 = cv2.threshold(src, 70, 255, cv2.THRESH_BINARY)
        cv2.imshow("Thresh2", src_thresh2)

        # Apply Erosion
        src = src_thresh
        kernel = np.ones((5, 5), np.uint8)
        iterations = 1
        img_erosion = cv2.erode(src, kernel, iterations=iterations+1)
        cv2.imshow('Erosion2', img_erosion)
        # Apply Dilation
        src = img_erosion
        img_dilation = cv2.dilate(src, kernel, iterations=iterations)
        cv2.imshow('Dilation2', img_dilation)

        # Apply Edgedetection
        # edges = cv2.Canny(src, 50, 70)
        # cv2.imshow('Edges', edges)

        # Find contours
        src = img_dilation
        contours, _ = cv2.findContours(src, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key = cv2.contourArea)
        # Find approx rectangle contours
        for c in contours[-5:]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            edges = len(approx)
            # if edges >= 3 and edges <= 5:
            cv2.drawContours(orig, [c], 0, (0, 255, 0), 2)
        cv2.imshow("Contours", orig)

    def threshold(self, src):
        orig = src

        ##################
        # begin threshold
        ##################

        # Threshold the HSV image to get only white colors
        src = orig
        lower_white = np.array([0,0,0], dtype=np.uint8)
        upper_white = np.array([150,100,255], dtype=np.uint8)
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_white, upper_white)
        thresh_hsv = cv2.bitwise_and(src, src, mask=mask)
        cv2.imshow("hsv", hsv)
        # cv2.imshow('Thresh HSV', np.hstack([hsv, thresh_hsv]))



        src = orig
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        kernel = np.ones((5, 5))
        iterations = 2
        # morph = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        # morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=iterations)
        morph = src
        morph = cv2.erode(morph, kernel, iterations=iterations)
        morph = cv2.dilate(morph, kernel, iterations=iterations)
        cv2.imshow('morph', morph)

        src = morph
        _, thresh = cv2.threshold(src, 128, 0, cv2.THRESH_TOZERO)
        cv2.imshow('Thresh', thresh)

        src = thresh
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        cv2.imshow("hsv", hsv)

        src = morph
        channels = np.split(np.asarray(src), 3, axis=2)
        height, width, _ = channels[0].shape
        for i in range(3):
            _, channel = cv2.threshold(channels[0], 128, 0, cv2.THRESH_TOZERO)
            channels[i] = np.reshape(channel, newshape=(height, width, 1))
        # merge the channels
        channels = np.concatenate(channels, axis=2)
        cv2.imshow("channels", channels)


        # Remove reflection
        src = orig
        # _, src_thresh = cv2.threshold(src, 200, 0, cv2.THRESH_TRUNC)
        _, mask = cv2.threshold(src, 200, 128, cv2.THRESH_BINARY)
        _, src_thresh = cv2.threshold(src, 50, 0, cv2.THRESH_TRUNC)
        # src_thresh = cv2.bitwise_and(src, mask)
        # cv2.imshow("Reflection2", mask)
        # cv2.imshow("Reflection", src_thresh)

    def morph(self, src):
        #############
        # begin morph
        #############

        morph = src.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=3)
        cv2.imshow("close", morph)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        cv2.imshow("open", morph)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        # take morphological gradient
        gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)
        cv2.imshow("grad", gradient_image)

        # split the gradient image into channels
        image_channels = np.split(np.asarray(gradient_image), 3, axis=2)
        channel_height, channel_width, _ = image_channels[0].shape
        # apply Otsu threshold to each channel
        for i in range(0, 3):
            _, image_channels[i] = cv2.threshold(~image_channels[i], 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
            image_channels[i] = np.reshape(image_channels[i], newshape=(channel_height, channel_width, 1))
        # merge the channels
        image_channels = np.concatenate((image_channels[0], image_channels[1], image_channels[2]), axis=2)

        # save the denoised image
        cv2.imshow("morph", image_channels)

    def contourcrop(self, src):
        ##############
        # contour crop
        ##############

        # contours = filter(self.filterLargeContour, contours)

        # Create mask where white is what we want, black otherwise
        # mask = np.zeros_like(src_thresh_inv1)
        # Draw filled contour in mask
        # cv2.drawContours(mask, [c], 0, 255, -1)
        # cv2.imshow("Mask", mask)

        # x, y, h, w = cv2.boundingRect(c)
        # lines1 = hough(l, mask)
        # findRectDivideConquer(src1, lines1)
        # cv2.imshow("Lines1", src1)

        # # Extract out the object and place into output image
        # out = np.zeros_like(src_thresh_inv1)
        # out[mask == 255] = src_thresh_inv1[mask == 255]

        # # Now crop
        # print(mask)
        # (y, x) = np.where(mask == 255)
        # (topy, topx) = (np.min(y), np.min(x))
        # (bottomy, bottomx) = (np.max(y), np.max(x))
        # out = out[topy:bottomy+1, topx:bottomx+1]

        # # Show the output image
        # cv2.imshow('Output', out)

        # c2 = c[:,0,:]
        # x, y, h, w = cv2.boundingRect(c)
        # c3 = np.array([[y+h, x], [y+h, x+w], [y, x+w], [y, x]])
        # cv2.rectangle(src1, (y+h, x), (y, x+w), (255,0,0), 2)
        # cv2.imshow("Contours", src1)
        # # print("cr3", c3)
        # src_cropped1 = four_point_transform(src1, c3, self.params['width'], self.params['height'])
        # cv2.imshow("Cropped", src_cropped1)
        # # raw_input()
        pass

    def updatepoints(self):
        # # Update cropping points
        # if self.points is not None:
        #     if np.allclose(self.points, points, atol=self.params['closeness']):
        #         print("close to pts")
        #         self.new_points = None
        #         self.new_seen = 0
        #     else:
        #         if self.new_points is not None:
        #             if np.allclose(self.new_points, points, atol=self.params['closeness']):
        #                 if self.new_seen == 5:
        #                     print("### switch to newpts ###")
        #                     self.points = self.new_points
        #                     self.new_points = None
        #                     self.new_seen = 0
        #                 else:
        #                     print("close to newpts")
        #                     self.new_seen += 1
        #             else:
        #                 print("update newpts")
        #                 self.new_points = points
        #                 self.new_seen = 1
        #         else:
        #             print("init newpts")
        #             self.new_points = points
        #             self.new_seen = 1
        # else:
        #     print("init _pts")
        #     self.points = points
        # self.points = points

        # print("points", points)
        pass

    def pipelineB(self, orig):
        # Apply blurring to sharpen
        # src = orig.copy()
        # src_blur = cv2.GaussianBlur(src, (5, 5), 3, borderType=cv2.BORDER_ISOLATED)
        # src_sharp1 = cv2.addWeighted(src, 1.5, src_blur, -0.7, 0)
        # cv2.imshow("Sharp", src_sharp)

        # Apply sharpening filter
        src = orig.copy()
        kernel = np.array([[-1,-1,-1],
                           [-1, 9,-1],
                           [-1,-1,-1]])
        src_sharp = cv2.filter2D(src, -1, kernel)
        # cv2.imshow("Sharp", src_sharp)


        # Apply smoothening preserving edge
        # src = src_sharp
        # src_filter = cv2.edgePreservingFilter(src, flags=cv2.RECURS_FILTER, sigma_s=60, sigma_r=0.4)
        # cv2.imshow("Filter", src_filter)

        src = src_sharp
        b = []
        g = []
        r = []
        for h in range(100, 140): # 480
            for w in range(300, 340): # 640
                b.append(src[h][w][0])
                g.append(src[h][w][1])
                r.append(src[h][w][2])
                for k in range(3):
                    src[h][w][k] = 255
        avg_b = sum(b)/len(b)
        avg_g = sum(g)/len(g)
        avg_r = sum(r)/len(r)
        delta = 60
        m = 1.5
        lower = np.array([max(0, avg_b-delta), max(0, avg_g-delta), max(0, avg_r-delta)], dtype = "uint8")
        upper = np.array([min(255, avg_b+m*delta), min(255, avg_b+m*delta), min(255, avg_b+m*delta)], dtype = "uint8")
        mask = cv2.inRange(src, lower, upper)
        src_white = cv2.bitwise_and(src, src, mask = mask)
        # cv2.imshow('White', src_white)

        # Convert the color from BGR to Gray
        src = src_white
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Gray", src_gray)

        src = src_gray
        morph = src
        kernel = np.ones((3, 3), np.uint8)
        iterations = 2
        morph = cv2.erode(morph, kernel, iterations=iterations+1)
        morph = cv2.dilate(morph, kernel, iterations=iterations)
        # cv2.imshow('Morph Open', morph)

        # Apply smoothening preserving edge
        src = morph
        src_filter = cv2.edgePreservingFilter(src, flags=cv2.RECURS_FILTER, sigma_s=60, sigma_r=0.4)
        # cv2.imshow("Filter", src_filter)

        # Apply mean adaptive theshold
        src = src_filter
        src_thresh = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 7)
        # cv2.imshow("Thresh", src_thresh)

        src = src_thresh
        morph = src
        kernel = np.ones((3, 3), np.uint8)
        iterations = 1
        morph = cv2.erode(morph, kernel, iterations=iterations+1)
        morph = cv2.dilate(morph, kernel, iterations=iterations)
        # cv2.imshow('Morphe Open 2', morph)

        # Find contours
        src = morph
        src_ex = orig.copy()
        contours, _ = cv2.findContours(src, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key = cv2.contourArea)
        # Find approx rectangle contours
        for c in contours[-3:]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            area = cv2.contourArea(c)
            edges = len(approx)
            if edges >= 3 and edges <= 5 and area > 0.03 * 640 * 480 and area < 0.9 * 640 * 480:
                cv2.drawContours(orig, [c], 0, (0, 255, 0), 2)
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(src_ex, (x,y), (x+w,y+h), (255,0,0), 2)
            # else:
            #     cv2.drawContours(orig, [c], 0, (255, 0, 0), 2)
        # cv2.imshow("Contours", src_ex)

        show1 = [orig, src_ex]
        show2 = [src_gray, src_filter, src_thresh]
        cv2.imshow("proc", np.hstack(show2))
        cv2.imshow("result", np.hstack(show1))


    def extract_whiteboard(self, src, frame):
        # # Skip if not sampling frame
        # if frame % self.params['freq'] != 0:
        #     if self.points is not None:
        #         src_cropped = four_point_transform(src, self.points, self.params['width'], self.params['height'])
        #     else:
        #         src_cropped = src
        #     return src_cropped

        l = min(self.params['width'], self.params['height'])
        self.pipelineB(src.copy())

        # Crop the image
        # src_cropped = four_point_transform(src, self.points, self.params['width'], self.params['height'])
        return src
