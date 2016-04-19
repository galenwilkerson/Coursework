import MapReduce
import sys

"""
Join Example in the Simple Python MapReduce Framework
"""

mr = MapReduce.MapReduce()

# =============================
# Do not modify above this line

def mapper(record):
    # key: document identifier
    # value: document contents
    order_id = record[1]
    value = record

    # for each element in value
    # shuffle the value to machine handling this order_id
#    for w in value:
    mr.emit_intermediate(order_id, value)
    #mr.emit_intermediate(value, key)


def reducer(key, list_of_values):
    # key: word
    # value: list of occurrence counts
#    total = 0
#    for v in list_of_values:
#      total += v

    # join all pairs
    for i in range(len(list_of_values)):
        for j in range(i+1, len(list_of_values)):
            if list_of_values[i][0] != list_of_values[j][0]:
                temp = list(list_of_values[i])
                temp.extend(list_of_values[j])
#            mr.emit(list_of_values[i].extend(list_of_values[j]))
                mr.emit(temp)            
            
# Do not modify below this line
# =============================
if __name__ == '__main__':
  inputdata = open(sys.argv[1])
 # inputdata = open("data/records.json")
  mr.execute(inputdata, mapper, reducer)
