import re
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef str preprocess_sentences(str text):
    cdef str c
    cdef str __t = " "
    cdef int b
    text = re.sub(r'[\W\s\_0-9]+', __t, text)
    b = False
    cdef list new_text = list()
    for c in text:
        if c.islower():
            new_text.append(c)
            b = True
        elif c.istitle():
            if b:
                new_text.append(__t)
                new_text.append(c)
            else:
                new_text.append(c)
            b = False
        else:
            new_text.append(c)
            b = False
    return "".join(new_text)
