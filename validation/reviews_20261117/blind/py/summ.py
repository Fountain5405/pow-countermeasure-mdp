import json, sys
for line in open(sys.argv[1]):
    if not line.startswith("{"): continue
    o = json.loads(line); h = o["howard"]
    print(f"{o[chr(108)+chr(97)+chr(98)+chr(101)+chr(108)]:28s} n={o[chr(110)+chr(95)+chr(115)+chr(116)+chr(97)+chr(116)+chr(101)+chr(115)]:7d} HOWARD g={h[chr(103)]:.12e} g_stat={h[chr(103)+chr(95)+chr(115)+chr(116)+chr(97)+chr(116)+chr(105)+chr(111)+chr(110)+chr(97)+chr(114)+chr(121)]:.12e} gap={h[chr(98)+chr(101)+chr(108)+chr(108)+chr(109)+chr(97)+chr(110)+chr(95)+chr(103)+chr(97)+chr(112)+chr(95)+chr(97)+chr(108)+chr(108)]:.1e} it={h[chr(105)+chr(116)+chr(101)+chr(114)+chr(115)]}")
    for k, v in o.items():
        if k.startswith("pto"):
            if "illegal_reached" in v: print(f"   {k:22s} ILLEGAL policy"); continue
            print(f"   {k:22s} value={v[chr(118)+chr(97)+chr(108)+chr(117)+chr(101)]:.12e} g_stat={v[chr(103)+chr(95)+chr(115)+chr(116)+chr(97)+chr(116)+chr(105)+chr(111)+chr(110)+chr(97)+chr(114)+chr(121)]:.12e} gap={v[chr(98)+chr(101)+chr(108)+chr(108)+chr(109)+chr(97)+chr(110)+chr(95)+chr(103)+chr(97)+chr(112)+chr(95)+chr(97)+chr(108)+chr(108)]:.1e} impr={v[chr(115)+chr(116)+chr(97)+chr(116)+chr(101)+chr(115)+chr(95)+chr(105)+chr(109)+chr(112)+chr(114)+chr(111)+chr(118)+chr(97)+chr(98)+chr(108)+chr(101)+chr(95)+chr(114)+chr(101)+chr(97)+chr(99)+chr(104)]} poldiff={v[chr(112)+chr(111)+chr(108)+chr(105)+chr(99)+chr(121)+chr(95)+chr(100)+chr(105)+chr(102)+chr(102)+chr(95)+chr(118)+chr(115)+chr(95)+chr(104)+chr(111)+chr(119)+chr(97)+chr(114)+chr(100)]} it={v[chr(105)+chr(116)+chr(101)+chr(114)+chr(115)]}")
