import cv2 as cv
import numpy as np
import time
from cards import load_ranks, load_suits
from videostream import VideoStream
from frame import process_frame, find_cards, process_card, identify_card, draw_card_info

def main():
    videostream = VideoStream().start()
    time.sleep(1.0)
    
    ranks, suits = load_ranks(), load_suits()
    
    while True:
        frame = videostream.read()
        
        processed_frame = process_frame(frame)
        
        cnts_sort, cnt_is_card = find_cards(processed_frame)
        
        if len(cnts_sort) != 0:
            cards = []
            k = 0
            for i in range(len(cnts_sort)):
                if cnt_is_card[i] == 1:
                    cards.append(process_card(cnts_sort[i], frame))

                    cards[k].best_rank_match, cards[k].best_suit_match, cards[k].rank_diff, cards[k].suit_diff = identify_card(cards[k], ranks, suits)
                    
                    frame = draw_card_info(frame, cards[k])
                    k += 1

            if (len(cards) != 0):
                temp_cnts = []
                for i in range(len(cards)):
                    temp_cnts.append(cards[i].contour)
                cv.drawContours(frame, temp_cnts, -1, (255, 0, 0), 2)
            
        cv.imshow('Processed Frame',processed_frame)
        cv.imshow('Tracker',frame)
        
        if cv.waitKey(1) == ord('q'):
            print('Quitting...')
            break
        
    cv.destroyAllWindows()
    videostream.stop()
    
if __name__ == "__main__":
    main()