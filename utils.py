KEYWORD = "keyword"
LEFT_TOP = "lt"
RIGHT_BOTTOM = 'rb'
ID = 'id'

class FrameKeywordInfo:

    def __init__(self, inf):
        self.kw = inf[KEYWORD]
        self.lt = inf[LEFT_TOP]
        self.br = inf[RIGHT_BOTTOM]

def getTime(dif):
    if dif < 60:
        return f"{int(dif)}s"
    else:
        c = int(dif // 60)
        if c < 60:
            return f"{c}m {int(dif % 60)}s"
        else:
            h = int(c // 60)
            return f"{h}h {c % 60}m {int(dif % 3600)}s"


def calculate_remaining_time(p, t, ps):
    rem = (t - p) / ps 
    t = getTime(rem)
    return t
