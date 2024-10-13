import json
import uuid
from polygon import Polygon, Block
import pickle
import requests
import threading
import numpy as np
from numpy.linalg import norm
from sklearn.cluster import KMeans

def parse(string):
    try:
        j = json.loads(string)
    except: 
        return string
    for k in j:
        if type(j[k]) is str and j[k] != None:
            j[k] = parse(j[k])
    return j


def make_blocks() -> dict[str, Block]:
    with open("datafiles/usa.geojson", "r") as f:
        string = f.read()
        j = parse(string)
        features = j["features"] # type: ignore
        #for feat in features:
        blocks = {}
        sizes = set()
        for feat in features:
            coords = feat["geometry"]["coordinates"][0] # type: ignore
            if (len(coords[0]) != 2):
                continue
            props = feat["properties"] # type: ignore
            block = Block(Polygon(coords), props["POP20"]) # type: ignore
            blocks[block.polygon.uuid] = block
        return blocks
        



addresses = {}

def format_address(address : str, neighborhood : str):
    address += f", {neighborhood}, MA"
    formatted = address.replace(" ", "+").replace("&", "and").replace("\n", " ")  
    return formatted

def address_to_coords(address : str, neighborhood : str):
    print(address)
    formatted = format_address(address, neighborhood)
    if formatted in addresses:
        print(formatted)
        return
    # AIzaSyBuus780dlerTVuhGzjjq2Jg1HbzHECeZg
    try:
        res = requests.get(f"https://maps.googleapis.com/maps/api/geocode/json?address={formatted}&key=AIzaSyBuus780dlerTVuhGzjjq2Jg1HbzHECeZg")
        table = json.loads(res.text)
        pos = table["results"][0]["geometry"]["viewport"]
        addresses[formatted] = pos
        print(formatted)
    except:
        print(f"error {formatted}")
        return




import os 
import math

def make_addresses(lines, addresses):
    for l in lines:
        values = l.split(",")
        a = values[6].replace('"', "")
        n = values[5].replace('"', "")
        address_to_coords(a, n)


def get_addresses():
    global addresses
    if "addresses" in os.listdir("."):
        with open("addresses", "rb") as f:
            addresses = pickle.load(f)
    else:
        addresses = {}
    buff = 0 
    with open("../crime.csv", "r") as f:
        lines = f.readlines()
    line_groups = []
    for i in range(1, len(lines), math.floor(len(lines)/10)):
        line_groups.append(lines[i : i + math.floor(len(lines)/10)])
    threads = []
    for group in line_groups:
        threads.append(threading.Thread(target = make_addresses, args = (group, addresses)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    pickle.dump(addresses, open("addresses", "wb"))




def read_addresses():
    with open("addresses", "rb") as f:
        addresses = pickle.load(f)
    return addresses

def read_blocks():
    with open("blocks", "rb") as f:
        blocks = pickle.load(f)
    return blocks

addresses_to_block_id = {}

def get_polygons_in(formatted : str, blocks : dict[str, Block], addresses : dict) -> []:
    if formatted in addresses_to_block_id:
        return addresses_to_block_id[formatted]
    value = addresses[formatted]
    coords1 = (value["northeast"]["lng"], value["northeast"]["lat"], )
    coords2 = (value["southwest"]["lng"], value["southwest"]["lat"])
    coords = ((coords1[0] + coords2[0])/2.0, (coords1[1] + coords2[1])/2.0)
    out = []
    for poly_id in blocks:
        if coords in blocks[poly_id].polygon:
            out.append(blocks[poly_id].polygon.uuid)
    addresses_to_block_id[formatted] = out
    return out




    

def save_blocks_with_crimes():
    addresses = read_addresses()
    blocks = make_blocks()
    i = 0
    with open("../crime.csv", "r") as f:
        print("starting\n")
        lines = f.readlines()
        groups = []
        inc = math.floor(len(lines)/10)
        for n in range(0, len(lines), inc):
            groups.append(lines[n : n + inc])

        for l in lines:
            values = l.split(",")
            a = values[6].replace('"', "")
            n = values[5].replace('"', "")
            formatted = format_address(a, n)
            if formatted not in addresses:
                continue
            poly_ids = get_polygons_in(formatted, blocks, addresses)
            description = values[3].replace('"', "")
            for poly_id in poly_ids:
                blocks[poly_id].add_crime(description)
            i += 1
            print(i, end = "\r")
    
    pickle.dump(blocks, open('blocks', "wb"))
    
    


from openai import OpenAI
client = OpenAI()

def cosine_sim(v1, v2):
    return np.dot(v1,v2)/(norm(v1)*norm(v2))

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return np.array(client.embeddings.create(input = [text], model=model, dimensions = 256).data[0].embedding)


CRIME_RANK2 = "murder rape arson aggravated assault first degree kidnapping human trafficking"
CRIME_RANK1 = "shoplifting stealing grafitti accident larcenry possesion"
CRIME_RANK0 = "hugs smiles laughter joy good times clean fun smiley face"


def make_crime_embeddings():
    def inner(data2):
        global x
        for tup in data2:
            if tup[1] not in embeddings:
                embeddings[tup[1]] = get_embedding(tup[1])
    embeddings = load_embeddings()
    with open("add_crime_tups", "rb") as f:
        data = pickle.load(f)
    groups = []
    inc = math.floor(len(data)/20)
    for i in range(0, len(groups), inc):
        groups.append(data[i : i + inc])
    threads = []
    for group in groups:
        threads.append(threading.Thread(target = inner, args = (group)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    embeddings[CRIME_RANK2] = get_embedding(CRIME_RANK2)
    embeddings[CRIME_RANK1] = get_embedding(CRIME_RANK1)
    embeddings[CRIME_RANK0] = get_embedding(CRIME_RANK0)
    pickle.dump(embeddings, open("embeddings", "wb"))
    return embeddings


    
def get_crime_emebedding(crime : str, embeddings):
    if crime in embeddings:
        return embeddings[crime]
    else:
        return get_embedding(crime)



scores = {}
def crime_to_score(crime : str, embeddings):
    global scores
    embedding = get_crime_emebedding(crime, embeddings)
    if crime in scores:
        return scores[crime]
    rank2 = get_crime_emebedding(CRIME_RANK2, embeddings)
    rank1 = get_crime_emebedding(CRIME_RANK1, embeddings)
    rank0 = get_crime_emebedding(CRIME_RANK0, embeddings)

    sim2 = cosine_sim(embedding, rank2)
    sim1 = cosine_sim(embedding, rank1)
    sim0 = cosine_sim(embedding, rank0)

    sims = np.array([sim0, sim1, sim2])

    out = np.argmax(sims).item()
    scores[crime] = out
    return out

def load_embeddings():
    if "embeddings" in os.listdir("."):
        with open("embeddings", "rb") as f:
            return pickle.load(f)
    else:
        return {}
    

def score_region(crimes : list[str], population : int, embeddings) -> float:
    out_score = 0.0
    for crime in crimes:
        out_score += crime_to_score(crime, embeddings)
    if population == 0:
        return 0
    out_score /= float(population)
    return out_score


def coords4_to_lat_longs(coords : list[tuple[float, float]]):
    return [str(c) for c in [coords[0][1], coords[0][0], coords[1][1], coords[1][0], coords[2][1], coords[2][0], coords[3][1], coords[3][0]]]




def final_csv():
    def inner(poly_ids, blocks, out_data2):
        for poly_id in poly_ids:
            block = blocks[poly_id]
            if len(block.crimes) > 0:
                score = score_region(block.crimes, block.population, embeddings)
                simplified = block.polygon.simplify(4)
                coords4 = simplified.getCoords()
                out_line = [poly_id] + coords4_to_lat_longs(coords4) + [score]
                out_data2.append(out_line)
            print(f"{len(out_data2)}/{len(blocks)}", end = "\r")

    embeddings = make_crime_embeddings()
    blocks = read_blocks()
    # blocks should already have crimes saved
    out_data = []
    first_line = ["cluster_id"]
    for i in range(4):
        first_line.append(f"vertex{i}_lat")
        first_line.append(f"vertex{i}_lon")
    first_line.append("crime_score")
    out_data.append(first_line)
    n = 0
    groups = []
    poly_ids = [p for p in blocks]
    inc = math.floor(len(poly_ids) / 20)
    threads = []
    for x in range(0, len(poly_ids), inc):
        groups.append(poly_ids[x : x + inc])
    for group in groups:
        threads.append(threading.Thread(target = inner, args = (group, blocks, out_data)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()


    """
    # Now, reescale calculated crime scores to 5 clusters using k-means
    model = KMeans(n_clusters = 5)
    array = np.array([row[-1] for row in out_data[1:]]).reshape(-1,1)
    model.fit(array)
    labels = model.predict(array)
    print(labels)
    print(len(labels))
    print(len(out_data))
    for i in range(len(labels)):
        out_data[i + 1][-1] = str(labels[i] + 1)  
    out_str = ""
    for line in out_data:
        out_str += ",".join(line) + "\n"
    with open("final.csv", "w") as f:
        f.write(out_str)
    """



def final_csv_fast():
    def inner(poly_ids, blocks, out_data2):
        for poly_id in poly_ids:
            block = blocks[poly_id]
            if len(block.crimes) > 0:
                score = 1 if block.population == 0 else 1 + len(block.crimes) / (block.population ** 3)
                simplified = block.polygon.simplify(4)
                coords4 = simplified.getCoords()
                out_line = [poly_id] + coords4_to_lat_longs(coords4) + [score]
                out_data2.append(out_line)
    
            print(f"{len(out_data2)}/{len(blocks)}", end = "\r")

    blocks = read_blocks()
    # blocks should already have crimes saved

    out_data = []
    first_line = ["cluster_id"]
    for i in range(4):
        first_line.append(f"vertex{i}_lat")
        first_line.append(f"vertex{i}_lon")
    first_line.append("crime_score")
    out_data.append(first_line)
    n = 0
    groups = []
    poly_ids = [p for p in blocks]
    inc = math.floor(len(poly_ids) / 20)
    threads = []
    for x in range(0, len(poly_ids), inc):
        groups.append(poly_ids[x : x + inc])
    for group in groups:
        threads.append(threading.Thread(target = inner, args = (group, blocks, out_data)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()





    """
    model = KMeans(n_clusters = 5)
    array = np.array([row[-1] for row in out_data[1:]]).reshape(-1,1)
    model.fit(array)
    labels = model.predict(array)
    print(labels)
    print(len(labels))
    print(len(out_data))


    
    for i in range(len(labels)):
        out_data[i + 1][-1] = str(labels[i] + 1)  
    """
    out_data = [out_data[0]] + [row[0 : -1] + [row[-1]] for row in sorted(out_data[1:], key = lambda row : row[-1], reverse = True)]
    scores = np.array([row[-1] for row in out_data[1:]])
    # Heuristic for caluclating category
    # Top 2% = 5
    # Top 5% >= 4
    # Top 10% >= 3
    # Top 50% >= 2
    # Bottom 50% = 1
    threshold5 = len(out_data)/50
    threshold4 = len(out_data)/20
    threshold3 = len(out_data)/10
    threshold2 = len(out_data)/2
    def get_cat(i):
        if i < threshold5:
            return 5
        elif i < threshold4:
            return 4
        elif i < threshold3:
            return 3
        elif i < threshold2:
            return 2
        else:
            return 1
    new_data = [out_data[0]]
    for i in range(1,len(out_data)):
        new_row = out_data[i][1:-1] + [str(get_cat(i))]
        new_data.append(new_row)

    out_data = new_data
    out_str = ""
    for line in out_data:
        out_str += ",".join(line) + "\n"
    out_str = ""
    for line in out_data:
        print(line)
        out_str += ",".join(line) + "\n"
    with open("datafiles/final_fast.csv", "w") as f:
        f.write(out_str)


addresses = {}
def process_boston_add_group(data):
    global addresses
    for d in data:
        if d["address"].strip() != "":
            address_to_coords(d["address"], d["neighborhood"])

def read_boston_crimes():
    global addresses 
    addresses = read_addresses()
    print("BEFORE", len(addresses))
    data = []
    with open("datafiles/boston_crime.csv", "r") as f:
        lines = f.readlines()
        for l in lines[1:]:
            l = l.split(",")
            data.append({"address": l[5], "description": l[4].casefold(), "neighborhood": l[-1]})
    groups = []
    inc = math.floor(len(data)/10)
    for i in range(0, len(data), inc):
        groups.append(data[i : i + inc])
    threads = []
    for g in groups:
        threads.append(threading.Thread(target = process_boston_add_group, args = (g,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        

    pickle.dump(addresses, open("addresses", "wb"))
    print("AFTER", len(addresses))

def make_full_crimes_tups():
    addresses = read_addresses()
    add_crime_tups = []
    with open("datafiles/boston_crime.csv", "r") as f:
        data = []
        lines = f.readlines()
        for l in lines[1:]:
            l = l.split(",")
            data.append({"address": l[5], "description": l[4].casefold().replace('"', ""), "neighborhood": l[-1]})
        for d in data:
            formatted = format_address(d["address"], d["neighborhood"])
            add_crime_tups.append({"formatted": formatted, "crime": d["description"]})
            
    with open("datafiles/crime.csv") as f:
        lines = f.readlines()
        for l in lines[1:]:
            values = l.split(",")
            a = values[6].replace('"', "")
            n = values[5].replace('"', "")
            formatted = format_address(a, n)
            description = values[3].replace('"', "")
            add_crime_tups.append({"formatted": formatted, "crime": description})
    pickle.dump(add_crime_tups, open("add_crime_tups", "wb"))



def save_blocks_with_crimes_full():
    def inner(data2):
        for d in data2:
            print(d)
            if d["formatted"] not in addresses or d["formatted"] == ",+ ,+MA":
                continue
            poly_ids = get_polygons_in(d["formatted"], blocks, addresses)
            description = d["crime"]
            for poly_id in poly_ids:
                blocks[poly_id].add_crime(description)

    addresses = read_addresses()
    blocks = make_blocks()
    i = 0
    with open("add_crime_tups", "rb") as f:
        print("starting\n")
        data = pickle.load(f)
        groups = []
        inc = math.floor(len(data)/10)
        for n in range(0, len(data), inc):
            groups.append(data[n : n + inc])
        threads = []
        for group in groups:
            threads.append(threading.Thread(target = inner, args = (group,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    
    pickle.dump(blocks, open('blocks', "wb"))


final_csv_fast()