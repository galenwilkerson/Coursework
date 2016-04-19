import MapReduce
import sys

"""
asymmetric Friend Count Example in the Simple Python MapReduce Framework
"""

mr = MapReduce.MapReduce()

# =============================
# Do not modify above this line

def mapper(record):
    
    # dnaName: document identifier
    # dnaString: the actual DNA code
    dnaName = record[0]    
    dnaString = record[1]
    
    # trim off last 10 chars
    dnaStringTrimmed = dnaString[0:len(dnaString) - 10]
    
    # insert into dictionary with value as key and key as value,
    # to avoid repeats
    mr.emit_intermediate(dnaStringTrimmed, dnaName)


def reducer(dnaStringTrimmed, list_of_dnaNames):

    # HMM, this should work...
    # check the length of the values this key (DNA string) point to in dictionary
    if (len(list_of_dnaNames) == 1):
        mr.emit((dnaStringTrimmed))#, list_of_values))

# Do not modify below this line
# =============================
if __name__ == '__main__':
  inputdata = open(sys.argv[1])
  #inputdata = open("data/dna.json")
  mr.execute(inputdata, mapper, reducer)
