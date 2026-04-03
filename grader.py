def grade_easy(score):
    return min(score / 10.0, 1.0)

def grade_medium(score):
    return min(score / 20.0, 1.0)

def grade_hard(score):
    return min(score / 30.0, 1.0)