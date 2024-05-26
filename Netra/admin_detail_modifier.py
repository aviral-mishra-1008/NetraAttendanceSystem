import pickle 

l = [('20225017',False),('20225022',False)]

with open('AttendanceSystem\\Netra\\prof_details.pkl','wb') as f:
    pickle.dump(l,f)

