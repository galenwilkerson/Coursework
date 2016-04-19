import json

"""
class to pretend map-reduce is actually distributing jobs across computers

"""
class MapReduce:
    def __init__(self):
        self.intermediate = {}
        self.result = []

    # just append the value to the key's entry in the dictionary
    def emit_intermediate(self, key, value):
        self.intermediate.setdefault(key, [])
        self.intermediate[key].append(value)

    def emit(self, value):
        self.result.append(value) 

    # map the data, then reduce it
    def execute(self, data, mapper, reducer):
        for line in data:
            record = json.loads(line)
            mapper(record)

        for key in self.intermediate:
            reducer(key, self.intermediate[key])

        #jenc = json.JSONEncoder(encoding='latin-1')
        jenc = json.JSONEncoder()
        for item in self.result:
            print jenc.encode(item)
