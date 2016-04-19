import MapReduce
import sys

"""
Word Count Example in the Simple Python MapReduce Framework
"""

mr = MapReduce.MapReduce()

# =============================
# Do not modify above this line

# bad style, assume matrix size
matrixSize = 10


def mapper(record):
    # key: document identifier
    # value: document contents


    matrixName = record[0]
    row = record[1]
    col = record[2]
    val = record[3]
     
    # emit cross product of rows of A, columns of B
    # one cross-product per emit
    if (matrixName == "a"):
        # emit copies of row (ideally one per column of B)
        for i in range(matrixSize):
            mr.emit_intermediate((row, i), record)
        
    if (matrixName == "b"):
        # emit copies of B
        for j in range(matrixSize):
            mr.emit_intermediate((j, col), record) 
            
def reducer(key, list_of_values):
 
    aVals = {}
    bVals = {}
    
    # get all a's, get all b's, insert into dictionary using k (the col of A and the row of B)
    for elt in list_of_values:
        if(elt[0] == "a"):
            aVals[elt[2]] = elt[3]
        else:
            bVals[elt[1]] = elt[3]

    if len(aVals) == 0 or len(bVals) == 0:
        return
 
    dotProd = 0
    for k in range(matrixSize):
        try:
            dotProd += aVals[k] * bVals[k]
        except:  # key not found
            continue            

    mr.emit((key[0], key[1], dotProd))

# Do not modify below this line
# =============================
if __name__ == '__main__':
  inputdata = open(sys.argv[1])
  #inputdata = open("data/matrix.json")
  mr.execute(inputdata, mapper, reducer)
