'''

    Author: Kartik Verma
    Created On: 07/03/2020
    Github link: http://github.com/vermakartik

    Please cite the authors of the code properly if you use the program or parts of it in your program.
    Using someone elses code or pushing it to your github, removing original authors name without proper citation,
    shows that you are a dumb asshole and a "code bandit"
    who does not know the decorum of writing quality code and crediting the originality. 

    The parts of code used for getting text bound boxes are taken from the following website 
    https://www.learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/

    The program uses pytesseract 3.02 which can be downloaded from the following link
    https://sourceforge.net/projects/tesseract-ocr-alt/files/tesseract-ocr-setup-3.02.02.exe/download

    
'''

import cv2
import numpy as np
import pytesseract as pst
import argparse
import matplotlib.pyplot as plt
import math
import logging
import threading
from collections import deque
import time
import utils as u

DEBUGING_MODE = False

IGNORE_MATCHED = 0x1
N_IGNORE_MATCHED = 0x2
NEW_MATCHED = 0x4

class Colliders:

    def __init__(
        self,
        shp=None,
        sample_image=None,
        path_to_east_net = "./frozen_east_text_detection.pb",
        pst_installation_path = r'E:/Tesseract-OCR/tesseract',
        size = (320, 320),
        threshold = 0.5,
        nms_thresh = 0.4,
        match_rate = 0.2,
        ignorance_rate = 10,
        ignore_bbs = 2
    ):

        self.sample_image = sample_image
        self.size = size
        self.shp = shp
        self.path_to_east_net = path_to_east_net
        self.pst_installation_path = pst_installation_path

        self.threshold = threshold
        self.nms_thresh = nms_thresh
        self.match_rate = match_rate

        self.__initial_conf__()

        self.keywords_to_look = self.__get_reference_samples__()

        self.prev_frame = None
        self.ignorance_rate = ignorance_rate
        self.ignore_bbs = ignore_bbs

    def __convert_format(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __get_reference_samples__(self):
        print(f"Sample Path: {self.sample_image}")
        
        img = self.__convert_format(cv2.imread(self.sample_image))
        
        logging.log(logging.DEBUG, f"Shape of Image: {img.shape} | Size: {self.size}.".encode('utf-8'))
        self.rt = (1.0, 1.0)
        logging.log(logging.DEBUG, f"Ratio: {self.rt}".encode('utf-8'))

        if DEBUGING_MODE:
            plt.imshow(img)
            plt.show()

        ks, _ = self.get_keywords(img, True)
        logging.log(logging.INFO, f"Got Keywords From Sample: {[x.kw for x in ks]}".encode('utf-8'))
        return ks

    def __initial_conf__(self):

        self.east_net = cv2.dnn.readNet(self.path_to_east_net)
        self.output_layers = []
        self.output_layers.append("feature_fusion/Conv_7/Sigmoid")
        self.output_layers.append("feature_fusion/concat_3")
        pst.pytesseract.tesseract_cmd = self.pst_installation_path

    def __get_center(self, f):

        shp = f.shape
        mnInd = 0 if shp[0] < shp[1] else 1
        mn = shp[mnInd]
        mx = shp[1 - mnInd]
        cmx = int(mx / 2)
        ons = int(mn / 2)
        if mnInd == 0:
            lr, lc = (0, cmx - ons)
            br, bc = (mn - 1, cmx + ons)
        else:
            lr, lc = (cmx - ons, 0)
            br, bc = (cmx + ons, mn + 1)
        
        return cv2.resize(f[lr:br, lc:bc, :], self.size)

    def __get_keywords(self, frame):

        # Fixed according to EAST Algorithm
        SUBTRACTER = (123.68, 116.78, 103.94)

        xx = cv2.UMat(self.__get_center(frame))

        if DEBUGING_MODE:
            
            plt.imshow(xx)
            plt.show()

        blb = cv2.dnn.blobFromImage(xx, 1.0, self.size, SUBTRACTER, False, False)

        self.east_net.setInput(blb)
        o = self.east_net.forward(self.output_layers)
        
        return (o, xx)
        
    def __process_output(self, o, frame):
        
        scores = o[0]
        geometry = o[1]

        [boxes, confidences] = self.__decode(scores, geometry, self.threshold)
        indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, self.threshold, self.nms_thresh)

        return (indices, boxes)

    def __process_bounding_boxes(self, indices, boxes):
        vals = []
        for i in indices:
            # get 4 corners of the rotated rect
            vertices = cv2.boxPoints(boxes[i[0]])
            # print(vertices)
            xmin, ymin, xmax, ymax = 999999, 999999, -1, -1
            # scale the bounding box coordinates based on the respective ratios
            for j in range(4):
                vertices[j][0] *= self.rt[1]
                vertices[j][1] *= self.rt[0]

                xmin = max(0, min((xmin, vertices[j][0])))
                xmax = min(self.size[1], max((xmax, vertices[j][0])))
                ymin = max(0, min((ymin, vertices[j][1])))
                ymax = min(self.size[0], max((ymax, vertices[j][1])))

            vals.append((int(xmin), int(ymin), int(xmax), int(ymax)))

        return vals

    def has_box_moved(self, b1, b2):
        
        mx_d = max(
            abs(b1.lt[0] - b2.lt[0]),
            abs(b1.lt[1] - b2.lt[1]),
            abs(b1.br[0]) - b2.br[0],
            abs(b1.br[1] - b2.br[1])
        )
        
        if mx_d <= self.ignorance_rate:
            return False
        return True


    def get_keywords(self, frame, show_kw = False):
         
        (ks, xx) = self.__get_keywords(frame)
        (i, b) = self.__process_output(ks, xx)
        bbs = self.__process_bounding_boxes(i, b)
        if show_kw == True:
            kw = self.__get_keywords_from_frame__(xx, bbs)
            return (kw, xx)
        else:
            return (bbs, xx)

    def __get_keywords_from_frame__(self, frame, boxes, keep_empty = True):
        vals = []

        if DEBUGING_MODE: 
            plt.imshow(frame)
            plt.show()
        
        for b in boxes:
                        
            if DEBUGING_MODE:
                plt.imshow(frame.get()[b[1]:b[3], b[0]:b[2], :])
                plt.show()

            kws = pst.image_to_string(frame.get()[b[1]:b[3], b[0]:b[2], :])
            if len(kws) > 0 or keep_empty == True:
                vals.append(u.FrameKeywordInfo({u.KEYWORD: kws, u.LEFT_TOP: (b[0], b[2]), u.RIGHT_BOTTOM: (b[1], b[3])}))

        return vals

    def __evalute(self, ks):
        pc = 0
        matched_words = []
        logging.log(logging.DEBUG, f"Got Keywords: {[x.kw for x in ks]}".encode('utf-8'))
        tlk = [x.kw for x in self.keywords_to_look]
        for kws in ks:
            kw = kws.kw
            if len(kw) <= 0:
                continue
            if kw in tlk:
                matched_words.append(kw)
                pc+=1
        cmr = (pc / len(self.keywords_to_look)) >= self.match_rate
        return cmr

    def match(self, inpFrame):
        
        if self.prev_frame != None:
            bbs, xx = self.get_keywords(inpFrame, False)
            dc = abs(len(bbs) - len(self.prev_frame))
            logging.log(logging.DEBUG, f"Values: Cur: {len(bbs)} - Prev: {len(self.prev_frame)}")
            logging.log(logging.DEBUG, f"Using Prev Frame: {[x.kw for x in self.prev_frame]}")
            if len(self.prev_frame) == 0 and len(bbs) > 0:
                logging.log(logging.DEBUG, f"Using New Frame Since previous was empty.".encode('utf-8'))
                ks = self.__get_keywords_from_frame__(xx, bbs)
                self.prev_frame = ks
                ev = self.__evalute(ks)
                return (ev, ks, NEW_MATCHED)
            elif dc <= self.ignore_bbs:
                logging.log(logging.DEBUG, f"Ignoring Frame Since change from previous is <= Ignore Value: {self.ignore_bbs}".encode('utf-8'))
                return (False, None, IGNORE_MATCHED)
            else:
                logging.log(logging.DEBUG, f"Can't Ignore Frame Since change from previous is > Ignore Value: {self.ignore_bbs}".encode('utf-8'))
                ks = self.__get_keywords_from_frame__(xx, bbs)
                self.prev_frame = ks
                ev = self.__evalute(ks)
                return (ev, ks, N_IGNORE_MATCHED)
        else:
            logging.log(logging.DEBUG, f"Cannot find Prev Frame Getting New Frame...".encode('utf-8'))
            ks, xx = self.get_keywords(inpFrame, True)
            self.prev_frame = ks
            ev = self.__evalute(ks)
            return (ev, ks, NEW_MATCHED)
    
    def __decode(self, scores, geometry, scoreThresh):
        
        detections = []
        confidences = []

        ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
        assert len(scores.shape) == 4, "Incorrect dimensions of scores"
        assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
        assert scores.shape[0] == 1, "Invalid dimensions of scores"
        assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
        assert scores.shape[1] == 1, "Invalid dimensions of scores"
        assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
        assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
        assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
        height = scores.shape[2]
        width = scores.shape[3]
        for y in range(0, height):

            # Extract data from scores
            scoresData = scores[0][0][y]
            x0_data = geometry[0][0][y]
            x1_data = geometry[0][1][y]
            x2_data = geometry[0][2][y]
            x3_data = geometry[0][3][y]
            anglesData = geometry[0][4][y]
            for x in range(0, width):
                score = scoresData[x]

                # If score is lower than threshold score, move to next x
                if(score < scoreThresh):
                    continue

                # Calculate offset
                offsetX = x * 4.0
                offsetY = y * 4.0
                angle = anglesData[x]

                # Calculate cos and sin of angle
                cosA = math.cos(angle)
                sinA = math.sin(angle)
                h = x0_data[x] + x2_data[x]
                w = x1_data[x] + x3_data[x]

                # Calculate offset
                offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

                # Find points for rectangle
                p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
                p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
                center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
                detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
                confidences.append(float(score))

        # Return detections and confidences
        return [detections, confidences]

class MatchInfo:

    def __init__(self, frame, name):
        self.frame = frame
        self.name = name

class SaveFrame:

    def __init__(self, pref):
        self.q = deque()
        self.qCount = 0
        self.folder = pref

        self.lock = threading.Condition()

        self.th1 = threading.Thread(target=self.frame_saver)
    
    def appendFrame(self, frameInfo: MatchInfo):
        
        self.q.append(frameInfo)
        self.qCount += 1
        self.frame_saver()

    def stop_thread(self):
        self.done = True

    def safe(self, ifrm: MatchInfo):
        logging.log(logging.DEBUG, f"Saving Frame with name: {ifrm.name}...".encode('utf-8'))
        cv2.imwrite(f"{self.folder}/{ifrm.name}.jpg", ifrm.frame)
    
    def frame_saver(self):
   
        hasFrame = False
   
        if self.qCount == 0:
            logging.log(logging.DEBUG, "Waiting For frames...")
            return
        else:
            logging.log(logging.DEBUG, "Got Frame!")
            hasFrame = True
            self.qCount -= 1
        if hasFrame == True:
            v = self.q.popleft()
            self.safe(v)
   
    def run(self):
        self.done = False
        self.th1.start()
        self.th1.join()

class Streamer:

    def __init__(self, stream_path, skip_rate, ignore_count, out_folder, wait_time, **colliders_args):
        
        self.stream_path = stream_path

        self.__open_stream()
        self.cl = Colliders(self.shp, **colliders_args)
        self.sf = SaveFrame(out_folder)
        self.skip_frame_rate = skip_rate
        self.ignore_count = ignore_count
        self.wait_time = wait_time
        
    def __open_stream(self):
        self.stream = cv2.VideoCapture(self.stream_path)
        self.shp = (int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)))

    def stream_it(self):
        cnt = 0
        tft = self.stream.get(cv2.CAP_PROP_FRAME_COUNT)
        c = time.time()
        logging.log(logging.INFO, f"Total Frames to process: {tft}")
        df = 1 / 30
        prev_found = None
        crnt = -1
        cwait = -1
        btime = time.time()
        pc = 0
        print("Starting Now... [press q to exit anytime]")
        while True:

            _, f = self.stream.read()
     
            if _ == False:
                logging.log(logging.ERROR, "\nCannot Get Frame. Closing Now!")
                break

            cnt += 1
            pc+=1
            crnt = (crnt + 1) % (self.skip_frame_rate + 1)
            
            if crnt != 0:
                continue
            logging.log(logging.DEBUG, f"Sending Frame: {cnt} | cwait: {cwait}")
     
            mtch, pfo, reason = self.cl.match(f)
     
            if pfo != None:
                logging.log(logging.DEBUG, f"Match: {mtch} | {[x.kw for x in pfo]}".encode('utf-8'))
            
            if mtch:
                spand = False
                logging.log(logging.DEBUG, f"Print Got Match : {cnt}")
                if prev_found is None:
                    spand = True
                    logging.log(logging.DEBUG, "Spand is True | Since Prev Found is None")
                    prev_found = pfo
                    cwait += 1
                else:
                    df = abs(len(pfo) - len(prev_found))
                    if df > self.ignore_count:
                        spand = True
                        prev_found = pfo
                    else:
                        cwait += 1
                        spand = True
 
                    logging.log(logging.DEBUG, f"Spand is {spand} | Since Prev Found is There.".encode('utf-8'))
                    
                if spand == True and cwait == 0:  
                    logging.log(logging.DEBUG, f"Wait time Passed! Can Save now!".encode('utf-8'))
                    self.sf.appendFrame(MatchInfo(f, f"frame_match_{cnt}"))
                    
            elif cwait != -1 and reason == IGNORE_MATCHED:
                logging.log(logging.DEBUG, f"Waiting: {cwait}".encode('utf-8'))
                cwait = (cwait + 1) % (self.wait_time)
                if cwait == 0:
                    self.sf.appendFrame(MatchInfo(f, f"frame_match_{cnt}"))
            elif reason == NEW_MATCHED or reason == N_IGNORE_MATCHED:
                cwait = -1
                prev_found = None
                logging.log(logging.DEBUG, f"Resetting New Matched")

            cv2.imshow("Video", f)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nExiting...")
                break

            if time.time() - c > 1:
                c = time.time()
                df = c - btime
                e = u.getTime(df)
                r = u.calculate_remaining_time(cnt, tft, pc)
                pc = 0
                print(f"Frames Processed: {cnt}/{tft} | Time Elapsed: {e} | Remaining Time: {r}", end="\r")

    def run(self):
        
        self.stream_it()

        self.stream.release()
        cv2.destroyAllWindows()
        

if __name__ == "__main__":

    p = argparse.ArgumentParser(description="DeepScreeno: An AI Based Screen Matching System which Captures Screen shot every time a particular screen comes")
    p.add_argument("-p", "--video_path", type=str, required=True, help="Path to Video which is to be processed")
    p.add_argument('-spl', "--sample_image", type=str, required=True, help="Sample Image which is used as a reference")
    p.add_argument('-est', "--east_path", type=str, help="Path to east Model", default="./frozen_east_text_detection.pb")
    p.add_argument('-tst', "--teserract_path", type=str, help="Path to east Model", default='E:/Tesseract-OCR/tesseract')
    p.add_argument('-th', "--threshold", type=float, help="Threshold value for scores", default=0.5)
    p.add_argument('-nth', '--nms_threshold', type=float, help="NMS Threshold Value", default=0.4) 
    p.add_argument('-dim', '--dimension', type=int, help="Side for the square used for Video Matching", default=320)
    p.add_argument('-mr', '--match_rate', type=float, help="Threshold Match rate to use for keyword Matching", default=0.2)
    p.add_argument('-l', '--log_level', type=int, help='Logging level[Following Levels are defined: \n CRITIAL: 50 | ERROR: 40 | WARNING: 30 | INFO: 20 | DEBUG: 10 | NOT_SET: 0]\n', default=logging.INFO)
    p.add_argument('-dbg', "--debug", action="store_true", help="Shows Debugging Details if set to true")
    p.add_argument('-o', "--output_path", type=str, help="Output Folder Path", default="./output_folder/")
    p.add_argument('-sk', "--skip_frame_rate", type=int, help="Number of Frames to skip before moving to next frame", default=2)
    p.add_argument('-ig', '--ignore_count', type=int, help="Ignore if change in keywords <= ignore_count", default=2)
    p.add_argument('-rep', "--report", type=str, help="Report Generated", default=None)
    p.add_argument('-wt', "--wait", type=int, help="Wait frames before saving", default=2)
    p.add_argument('-ir', '--ignorance_rate', type=int, help="Ignore if Bounding Boxes has not moved beyond Ignorance Rate", default=10)
    p.add_argument('-bb', '--ignore_bbs', type=int, help="Ignore if number of bbs changed from prev_found <= ignore_bbs | Set to high to ignore More bounding Boxes", default=2)

    args = p.parse_args()

    if args.report is not None:
        logging.basicConfig(
            level=args.log_level,
            filename=args.report,
            format='%(name)s - %(levelname)s - %(message)s'
        )
    else:
        logging.basicConfig(
            level=args.log_level,
            format='%(name)s - %(levelname)s - %(message)s'
        )

    logging.log(logging.DEBUG, f"Arguments: {args}")
    print(f"Got Arguments: {args}")
    
    DEBUGING_MODE = args.debug

    s = Streamer(
        args.video_path,
        ignore_count=args.ignore_count,
        skip_rate=args.skip_frame_rate,
        out_folder=args.output_path,
        wait_time=args.wait,
        sample_image=args.sample_image,
        path_to_east_net=args.east_path,
        pst_installation_path=args.teserract_path,
        size=(args.dimension, args.dimension),
        threshold=args.threshold,
        nms_thresh=args.nms_threshold,
        match_rate=args.match_rate
    )   
    s.run()



# shp,
#         sampleImage,
#         path_to_east_net = "./frozen_east_text_detection.pb",
#         pst_installation_path = r'E:/Tesseract-OCR/tesseract',
#         size = (320, 320),
#         threshold = 0.5,
#         nms_thresh = 0.4,
#         match_rate = 0.2
#         ignorance_rate = 10,
#         ignore_bbs = 2

