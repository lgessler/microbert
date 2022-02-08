"""
This script assumes the current directory contains the unzipped Greek and Latin Perseus raw XML data from
https://www.perseus.tufts.edu/hopper/opensource/downloads/texts/hopper-texts-GreekRoman.tar.gz

It will output plain text data approximating basic tokenization in one 'sentence' per line into two folders under
_perseus_out/ called greek/ and latin/, one document per input XML file.
"""

import html
import io
import os
import re
import sys
from glob import glob

import betacode.conv

ent_dict = html.entities.html5
ent_dict = {k:v for k, v in ent_dict.items() if ";" in k}
perseus_root = "."
out_root = "_perseus_out" + os.sep

files = glob(perseus_root + os.sep + "**" + os.sep + "*.xml",recursive=True)
files = [f for f in files if out_root not in f]
files = [f for f in files if f.endswith("_gk.xml") or f.endswith("_lat.xml")]

sent_end_tags = ["speaker","p","div1","div2"]
sent_start_tags = "<(" + "|".join(sent_end_tags) + "[^<>]*>"
sent_end_tags = "</(" + "|".join(sent_end_tags) + ")>"

# Unknown entities
additional = {"&stigma;":"Ϛ","&ldsqb;":"[","&rdsqb;":"]"}
remaining = {'&lpress;', '&open;', '&x1000;', '&responsibility;', '&close;', '&rpress;', '&stigma;', '&c;'}
ent_dict.update(additional)

spaces = {"greek":0,"latin":0}

for file_ in files:
    text = io.open(file_,encoding="utf8").read().replace("\r","")
    if file_.endswith("_lat.xml"):  # Latin original
        out_dir = out_root + "latin" + os.sep
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_handle = os.path.basename(file_).replace(".xml",".txt")
        greek = False
    elif file_.endswith("_gk.xml"):
        out_dir = out_root + "greek" + os.sep
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_handle = os.path.basename(file_).replace(".xml",".txt")
        greek = True
    else:
        continue

    sys.stderr.write("o processing " + os.path.basename(file_) + "\n")

    #text = re.sub(r'.*(<body>.*</body>).*',r'\1',text,count=1,flags=re.DOTALL)
    text = text[text.index("<body"):]
    text = text.replace("⌏","").replace("⌎","")
    text = re.sub(sent_end_tags,'%%%NEWLINE%%%',text)
    text = re.sub(r'([^\s]{2}[!?.;]+) ',r'\1%%%NEWLINE%%% ',text)
    text = text.replace("%%%NEWLINE%%%","\n")
    text = re.sub(r' +',' ',text)
    text = re.sub(r'<!--[^-]*?-->','',text, flags=re.DOTALL)
    text = re.sub(r'<[^<>]*?>',r'',text, flags=re.DOTALL)

    for ent in ent_dict:
        find = "&" + ent if not ent.startswith("&") else ent
        text = text.replace(find,ent_dict[ent])

    if greek:
        text = betacode.conv.beta_to_uni(text)

    output = []
    lines = text.split("\n")
    lines = [l for l in lines if len(l.strip())>0]
    for l, line in enumerate(lines):
        if len(line.strip()) == 0:
            continue
        next_line = "A" if l >= len(lines) - 1 else lines[l+1]
        if next_line == "":
            next_line = "A"
        line = line.strip() + " "
        if line.endswith(". ") and not line.endswith(" . "):
            line = line[:-2] + " . "
        line = line.replace(","," , ").replace("?"," ? ").replace(";"," ; ").replace("!"," ! ").replace(":"," : ").replace("·"," · ")
        line = line.replace("  "," ")
        if line[-2] not in [".","!","?",";","·",":"] and next_line[0].upper() != next_line[0]:  # Run on sentence across lines
            line += "%%RMNEWLINE%%"
        output.append(line)

    text = "\n".join(output)
    text = text.replace("%%RMNEWLINE%%\n","").strip().replace("  "," ") + "\n"
    if greek:
        spaces["greek"] += text.count(" ")
    else:
        spaces["latin"] += text.count(" ")

    with io.open(out_dir + out_handle,'w',encoding="utf8",newline="\n") as f:
        f.write(text)

print(spaces)