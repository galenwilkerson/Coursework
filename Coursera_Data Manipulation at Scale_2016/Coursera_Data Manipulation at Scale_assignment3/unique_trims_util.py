# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 09:28:34 2016

@author: username

utility to generate simpler test data

read in each dna, output last 20 chars

"""
import json

data = open("data/dna.json")
outData = open("data/test_dna.json", "w")

for line in data:
    record = json.loads(line)
    key = record[0]
    val = record[1]
    newVal = val[len(val)-50:len(val)]
    json.dump([key, newVal], outData)
    
data.close()
outData.close()