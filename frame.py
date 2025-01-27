import cv2 as cv
import numpy as np
import constants
from cards import QueryCard

def process_frame(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    
    img_w, img_h = np.shape(frame)[:2]
    bkg_level = gray[int(img_h / 100)][int(img_w / 2)]
    thresh_level = bkg_level + constants.BKG_THRESHOLD
    
    retval, thresh = cv.threshold(blur, thresh_level, 255, cv.THRESH_BINARY)
    
    return thresh

def find_cards(thresh_frame):
    cnts,hier = cv.findContours(thresh_frame,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda i : cv.contourArea(cnts[i]),reverse=True)
    
    if len(cnts) == 0:
        return [], []
    
    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts),dtype=int)
    
    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])
        
    for i in range(len(cnts_sort)):
        size = cv.contourArea(cnts_sort[i])
        peri = cv.arcLength(cnts_sort[i], True)
        approx = cv.approxPolyDP(cnts_sort[i], 0.01 * peri, True)
        
        if ((size < constants.CARD_MAX_AREA) and (size > constants.CARD_MIN_AREA) and (hier_sort[i][3] == -1) and (len(approx) == 4)):
            cnt_is_card[i] = 1
    
    return cnts_sort, cnt_is_card

def process_card(contour, frame):
    q_card = QueryCard(contour)
    
    perimeter = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.01 * perimeter, True)
    pts = np.float32(approx)
    q_card.corner_pts = pts
    
    x, y, w, h = cv.boundingRect(contour)
    q_card.width, q_card.height = w, h
    
    average = np.sum(pts, axis=0) / len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    q_card.center = [cent_x, cent_y]
    
    q_card.warp = flatten_card(frame, pts, w, h)
    
    card = q_card.warp[0:constants.CARD_CORNER_HEIGHT, 0:constants.CARD_CORNER_WIDTH]
    card = cv.resize(card, (0,0), fx=4, fy=4)
    
    white_level = card[15,int((constants.CARD_CORNER_WIDTH * 4) / 2)]
    thresh_level = white_level - constants.CARD_THRESHHOLD
    if thresh_level <= 0:
        thresh_level = 1
    retval, card_thresh = cv.threshold(card, thresh_level, 255, cv.THRESH_BINARY_INV)
    
    card_rank = card_thresh[20:185, 0:128]
    card_suit = card_thresh[186:336, 0:128]
    
    card_rank_cnts, hier = cv.findContours(card_rank, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    card_rank_cnts = sorted(card_rank_cnts, key=cv.contourArea, reverse=True)
    if len(card_rank_cnts) != 0:
        xr, yr, wr, hr = cv.boundingRect(card_rank_cnts[0])
        card_rank_roi = card_rank[yr:yr+hr, xr:xr+wr]
        card_rank_sized = cv.resize(card_rank_roi, (constants.RANK_WIDTH,constants.RANK_HEIGHT), 0, 0)
        q_card.rank_img = card_rank_sized
        
    card_suit_cnts, hier = cv.findContours(card_suit, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    card_suit_cnts = sorted(card_suit_cnts, key=cv.contourArea, reverse=True)
    if len(card_rank_cnts) != 0:
        xs, ys, ws, hs = cv.boundingRect(card_suit_cnts[0])
        card_suit_roi = card_suit[ys:ys+hs, xs:xs+ws]
        card_suit_sized = cv.resize(card_suit_roi, (constants.SUIT_WIDTH,constants.SUIT_HEIGHT), 0, 0)
        q_card.suit_img = card_suit_sized
    
    return q_card

def identify_card(q_card, ranks, suits):
    best_rank_match = ''
    best_suit_match = ''
    best_rank_diff = 1000000
    best_suit_diff = 1000000
    
    if (len(q_card.rank_img) != 0) and (len(q_card.suit_img) != 0):
        for rank in ranks:
            diff_img = cv.absdiff(q_card.rank_img, rank.img)
            rank_diff = int(np.sum(diff_img) / 255)
            if rank_diff < best_rank_diff:
                best_rank_diff_img = diff_img
                best_rank_diff = rank_diff
                best_rank_name = rank.name
        
        for suit in suits:
            diff_img = cv.absdiff(q_card.suit_img, suit.img)
            suit_diff = int(np.sum(diff_img) / 255)
            if suit_diff < best_suit_diff:
                best_suit_diff_img = diff_img
                best_suit_diff = suit_diff
                best_suit_name = suit.name
    
    if best_rank_diff < constants.RANK_DIFF_MAX:
        best_rank_match = best_rank_name
    
    if best_suit_diff < constants.SUIT_DIFF_MAX:
        best_suit_match = best_suit_name
    
    return best_rank_match, best_suit_match, best_rank_diff, best_suit_diff

def draw_card_info(frame, q_card):
    x = q_card.center[0]
    y = q_card.center[1]
    cv.circle(frame, (x, y), 5, (255, 0, 0), -1)
    
    rank_name = q_card.best_rank_match
    suit_name = q_card.best_suit_match
    
    cv.putText(frame, rank_name, (x - 60, y - 10), constants.FONT, 1, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(frame, rank_name, (x - 60, y - 10), constants.FONT, 1, (50, 200, 200), 2, cv.LINE_AA)
    
    cv.putText(frame, suit_name, (x - 60, y + 25), constants.FONT, 1, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(frame, suit_name, (x - 60, y + 25), constants.FONT, 1, (50, 200, 200), 2, cv.LINE_AA)
    
    return frame

def flatten_card(image, pts, w, h):
    temp_rect = np.zeros((4, 2), dtype="float32")
    
    s = np.sum(pts, axis=2)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=-1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    
    if w <= 0.8 * h:
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl
    
    if w >= 1.2 * h:
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br
    
    if w > 0.8 * h and w < 1.2 * h:
        if pts[1][0][1] <= pts[3][0][1]:
            temp_rect[0] = pts[1][0]
            temp_rect[1] = pts[0][0]
            temp_rect[2] = pts[3][0]
            temp_rect[3] = pts[2][0]
            
        if pts[1][0][1] > pts[3][0][1]:
            temp_rect[0] = pts[0][0]
            temp_rect[1] = pts[3][0]
            temp_rect[2] = pts[2][0]
            temp_rect[3] = pts[1][0]
    
    dst = np.array([[0, 0], [constants.CARD_MAX_WIDTH - 1, 0], [constants.CARD_MAX_WIDTH - 1, constants.CARD_MAX_HEIGHT - 1], [0, constants.CARD_MAX_HEIGHT - 1]], np.float32)
    dst = cv.getPerspectiveTransform(temp_rect, dst)
    warp = cv.warpPerspective(image, dst, (constants.CARD_MAX_WIDTH, constants.CARD_MAX_HEIGHT))
    warp = cv.cvtColor(warp, cv.COLOR_BGR2GRAY)
    
    return warp