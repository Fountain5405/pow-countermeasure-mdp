import re, sys, zipfile, xml.etree.ElementTree as ET
ns={"m":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
z=zipfile.ZipFile(sys.argv[1])
ss=[]
if "xl/sharedStrings.xml" in z.namelist():
    root=ET.fromstring(z.read("xl/sharedStrings.xml"))
    for si in root.findall("m:si",ns):
        ss.append("".join(t.text or "" for t in si.iter("{%s}t"%ns["m"])))
wb=ET.fromstring(z.read("xl/workbook.xml"))
rels=ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
rmap={r.get("Id"):r.get("Target") for r in rels}
for sh in wb.find("m:sheets",ns):
    rid=sh.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
    path="xl/"+rmap[rid]
    print("=== SHEET", sh.get("name"), path)
    root=ET.fromstring(z.read(path))
    for row in root.iter("{%s}row"%ns["m"]):
        cells=[]
        for c in row.findall("m:c",ns):
            v=c.find("m:v",ns); t=c.get("t")
            val="" if v is None else v.text
            if t=="s": val=ss[int(val)]
            cells.append(f"{c.get(chr(114))}={val}")
        print(" | ".join(cells))
