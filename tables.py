import json, numpy as np
def ci(v): v=np.asarray(v,float); return 100*v.mean(), 196*v.std(ddof=1)/np.sqrt(len(v))
NAMES=[("random","Random search"),("hill","Hill-climbing"),("eps","$\\varepsilon$-greedy"),("thompson","Thompson sampling"),
       ("ucb","UCB"),("ucb_screen","UCB, pre-screened"),("measure_once","Measure-once local search"),("twin_only","Twin only"),
       ("twin_validate","Simulate-then-validate"),("loop_blind","Loop, no pre-screening"),("loop_twinstart","Twin-seeded loop"),("loop","Loop")]
TUNED={"hill","eps","thompson","ucb","ucb_screen","twin_validate"}
A=json.load(open("results/policies_testbed.json")); B=json.load(open("results/policies_testbed_mid.json"))
print("% testbed table: shipped @100/200/300, found@300, belief@300, paired diff vs loop @300")
for k,n in NAMES:
    row=[]
    for b in ("100","200","300"):
        src=A
        if k in TUNED and np.mean(B[b][k]["shipped"])>np.mean(A[b][k]["shipped"]): src=B
        m,h=ci(src[b][k]["shipped"]) if k!="twin_only" else (100*src[b][k]["shipped"][0],0)
        row.append(f"${m:.1f}\\pm{h:.1f}$" if k!="twin_only" else f"${m:.1f}$")
    src=A if not (k in TUNED and np.mean(B["300"][k]["shipped"])>np.mean(A["300"][k]["shipped"])) else B
    f=ci(src["300"][k]["found"])[0]; be=np.nanmean(src["300"][k]["belief"])*100 if k!="twin_only" else float('nan')
    d=np.array(src["300"][k]["shipped"])-np.array(A["300"]["loop"]["shipped"]) if k not in("twin_only","loop") else None
    ds=f"${ci(d)[0]:+.1f}\\pm{ci(d)[1]:.1f}$" if d is not None else ("---" if k=="loop" else f"${100*(src['300'][k]['shipped'][0]-np.mean(A['300']['loop']['shipped'])):+.1f}$")
    print(f"{n} & "+" & ".join(row)+f" & ${f:.1f}$ & "+(f"${be:.1f}$" if k!="twin_only" else "---")+f" & {ds}\\\\")
Z=json.load(open("results/policies_zoo.json"))
print("% zoo table")
lo={lvl:{w:np.array(Z[lvl][w]["loop"]["shipped"]) for w in Z[lvl]} for lvl in Z}
for k,n in NAMES:
    cells=[]
    for lvl in ("tm","mid","high"):
        ws=list(Z[lvl]); m=[np.mean(Z[lvl][w][k]["shipped"]) for w in ws]
        dd=[np.mean(Z[lvl][w][k]["shipped"])-np.mean(Z[lvl][w]["loop"]["shipped"]) for w in ws]
        a=ci(m); b=ci(dd)
        cells.append(f"${a[0]:.1f}\\pm{a[1]:.1f}$"); cells.append("---" if k=="loop" else f"${b[0]:+.1f}\\pm{b[1]:.1f}$")
    print(n+" & "+" & ".join(cells)+"\\\\")
