import sys, time, os
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
import pandas as pd, torch
from nesy.reports import ReportZip
torch.set_num_threads(int(sys.argv[1]))
from radgraph import RadGraph
t=time.time(); rg=RadGraph(model_type="modern-radgraph-xl", batch_size=int(sys.argv[2]), cuda=-1); print("load s", round(time.time()-t,1), flush=True)
m=pd.read_parquet("/scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg/v2/data/manifests/mimic.parquet",columns=["subject_id","study_id"]).drop_duplicates("study_id").sample(128, random_state=0)
rz=ReportZip(); texts=[rz.sections(s,st).text for s,st in zip(m.subject_id,m.study_id)]
t=time.time(); out=rg(texts); dt=time.time()-t
print("reports/s", round(len(texts)/dt,2), "threads", sys.argv[1], "bs", sys.argv[2]); k=list(out)[0]; import json; print(json.dumps(out[k])[:800])
