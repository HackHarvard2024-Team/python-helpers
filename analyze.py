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
    with open("../usa.geojson", "r") as f:
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
    formatted = address.replace(" ", "+").replace("&", "and")   
    return formatted

def address_to_coords(address : str, neighborhood : str):
    formatted = format_address(address, neighborhood)
    if formatted in addresses:
        return addresses[formatted]
    # AIzaSyBuus780dlerTVuhGzjjq2Jg1HbzHECeZg
    res = requests.get(f"https://maps.googleapis.com/maps/api/geocode/json?address={formatted}&key=AIzaSyBuus780dlerTVuhGzjjq2Jg1HbzHECeZg")
    table = json.loads(res.text)
    pos = table["results"][0]["geometry"]["viewport"]
    addresses[formatted] = pos




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


a = read_addresses()



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
    
with open("blocks", "rb") as f:
    data = pickle.load(f)
    print(len(data))
    has = 0
    for poly_id in data:
        print(data[poly_id].crimes)
        if len(data[poly_id].crimes) > 0:
            has += 1
    print(has)
    


from openai import OpenAI
client = OpenAI()

def cosine_sim(v1, v2):
    return np.dot(v1,v2)/(norm(v1)*norm(v2))

def get_embedding(text, model="text-embedding-3-large"):
   text = text.replace("\n", " ")
   return np.array(client.embeddings.create(input = [text], model=model).data[0].embedding)


CRIME_RANK2 = "murder rape arson aggravated assault first degree kidnapping human trafficking"
CRIME_RANK1 = "shoplifting stealing grafitti accident larcenry possesion"
CRIME_RANK0 = "hugs smiles laughter joy good times clean fun smiley face"


def make_crime_embeddings():
    embeddings = {}
    with open("../crime.csv", "r") as f:
        lines = f.readlines()
        for l in lines:
            values = l.split(",")
            description = values[3].replace('"', "")
            if description not in embeddings:
                embeddings[description] = get_embedding(description)
    embeddings[CRIME_RANK2] = get_embedding(CRIME_RANK2)
    embeddings[CRIME_RANK1] = get_embedding(CRIME_RANK1)
    embeddings[CRIME_RANK0] = get_embedding(CRIME_RANK0)
    pickle.dump(embeddings, open("embeddings", "wb"))


    
def get_crime_emebedding(crime : str, embeddings):
    if crime in embeddings:
        return embeddings[crime]
    else:
        get_embedding(crime)



def crime_to_score(crime : str, embeddings):
    embedding = get_crime_emebedding(crime, embeddings)
    rank2 = get_crime_emebedding(CRIME_RANK2, embeddings)
    rank1 = get_crime_emebedding(CRIME_RANK1, embeddings)
    rank0 = get_crime_emebedding(CRIME_RANK0, embeddings)

    sim2 = cosine_sim(embedding, rank2)
    sim1 = cosine_sim(embedding, rank1)
    sim0 = cosine_sim(embedding, rank0)

    sims = np.array([sim0, sim1, sim2])

    return np.argmax(sims).item()

def load_embeddings():
    with open("embeddings", "rb") as f:
        return pickle.load(f)
    

def score_region(crimes : list[str], population : int, embeddings) -> float:
    out_score = 0.0
    for crime in crimes:
        out_score += crime_to_score(crime, embeddings)
    out_score /= float(population)
    return out_score


def coords4_to_lat_longs(coords : list[tuple[float, float]]):
    return [str(c) for c in [coords[0][1], coords[0][0], coords[1][1], coords[1][0], coords[2][1], coords[2][0], coords[3][1], coords[3][0]]]



def final_csv():
    if "embeddings" not in os.listdir("."):
        make_crime_embeddings()
    embeddings = load_embeddings()
    blocks = read_blocks()
    # blocks should already have crimes saved
    out_data = []
    first_line = ["cluster_id"]
    for i in range(4):
        first_line.append(f"vertex{i}_lat")
        first_line.append(f"vertex{i}_lon")
    first_line.append("crime_score")
    out_data.append(first_line)
    for poly_id in blocks:
        block = blocks[poly_id]
        if len(block.crimes) > 0:
            score = score_region(block.crimes, block.population, embeddings)
            simplified = block.polygon.simplify(4)
            coords4 = simplified.getCoords()
            out_line = [poly_id] + coords4_to_lat_longs(coords4) + [score]
            out_data.append(out_line)
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



final_csv()

