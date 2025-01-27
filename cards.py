import cv2 as cv

class QueryCard:
    def __init__(self, contour):
        self.contour = contour
        self.width, self.height = 0, 0
        self.corner_pts = []
        self.center = []
        self.warp = []
        self.rank_img = []
        self.suit_img = []
        self.rank = ''
        self.suit = ''
        
class Rank:
    def __init__(self):
        self.img = []
        self.name = ''
        
class Suit:
    def __init__(self):
        self.img = []
        self.name = ''

def load_ranks():
    ranks = []
    
    for i, rank in enumerate(['Ace', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King']):
        ranks.append(Rank())
        ranks[i].name = rank
        ranks[i].img = cv.imread(f'card_templates\\{rank}.jpg', cv.IMREAD_GRAYSCALE)
        
    return ranks

def load_suits():
    suits = []
    
    for i, suit in enumerate(['Spades', 'Diamonds', 'Clubs', 'Hearts']):
        suits.append(Suit())
        suits[i].name = suit
        suits[i].img = cv.imread(f'card_templates\\{suit}.jpg', cv.IMREAD_GRAYSCALE)
        
    return suits