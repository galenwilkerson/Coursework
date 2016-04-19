import MapReduce
import sys

"""
Word Count Example in the Simple Python MapReduce Framework
"""

mr = MapReduce.MapReduce()

# =============================
# Do not modify above this line

def mapper(record):
    # key: document identifier
    # value: document contents
    doc = record[0]
    value = record[1]
    words = value.split()

    # remove duplicates
    words = list(set(words))

    for w in words:
      # emit the word and the document it appeared in
      mr.emit_intermediate(w, doc)

def reducer(key, list_of_values):
    # key: word
    # value: list of occurrence counts
    total = 0
    
    # create a tuple containing (word, [list of document ids])
    mr.emit((key, list_of_values))

# Do not modify below this line
# =============================
if __name__ == '__main__':
#  inputdata = open(sys.argv[1])
  inputdata = open("data/books.json")
  mr.execute(inputdata, mapper, reducer)
