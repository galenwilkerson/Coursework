import MapReduce
import sys

"""
asymmetric Friend Count Example in the Simple Python MapReduce Framework
"""

mr = MapReduce.MapReduce()

# =============================
# Do not modify above this line

def mapper(record):
    # key: document identifier
    # value: document contents
    key = record[0]
    value = record[1]
    #words = value.split()
    #for w in words:
    mr.emit_intermediate((key, value), 1)
    mr.emit_intermediate((value, key), 1)
#    mr.emit_intermediate(value, key)

def reducer(key, list_of_values):
    # key: word
    # value: list of occurrence counts
#    total = 0
#    for v in list_of_values:
#      total += v
#    if list_of_values[0]
    if (len(list_of_values) != 2):
        mr.emit((key))#, list_of_values))

# Do not modify below this line
# =============================
if __name__ == '__main__':
  inputdata = open(sys.argv[1])
  #inputdata = open("data/friends.json")
  mr.execute(inputdata, mapper, reducer)
