import sys

abc = collections.namedtuple("abc", "a b c")  # Missing import

hi = ur"hello"                     # Illegal in Python 3
for i in xrange(2):                # Illegal in Python 3
    long_i = long(i)               # Illegal in Python 3
    print i                        # Illegal in Python 3
    if isinstance(i, basestring):  # Illegal in Python 3
        print(unicode(i))          # Illegal in Python 3
reload(sys)                        # Illegal in Python 3
sys.setdefaultencoding("utf-8")    # Illegal in Python 3

async def async_print(s):          # Illegal in Python 2
    print(s, end=" ")              # Illegal in Python 2
